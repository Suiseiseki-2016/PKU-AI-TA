"""
Crawler for PKU's custom homework system: bb-homeWorkCheck-BBLEARN.

Endpoints (all under /webapps/bb-homeWorkCheck-BBLEARN/homeWorkCheck/):

  getHomeWorkList.do?course_id=...
      → HTML page listing all assignments with gradeBookPK (numeric) and title

  getStudentWork.do?course_id=...&gradeBookPK=...&title=...&showAll=true
      → HTML page listing every submitted student:
          userId (real student number), name, filePk, attemptPk
        Links use CheckWork.do per student.

  CheckWork.do?course_id=...&gradeBookPK=...&userId=...&filePk=...&title=...&attemptPk=...
      → HTML page for one student's submission.
        JS embeds: var filePath = '/usr/local/blackboard/content/storage/pdf/{courseId}/{gradeBookPK}/{filePk}/{filename}'

  api/pdf.do?path={double_url_encoded_filePath}
      → Serves the PDF directly (application/pdf).
        NOTE: pass the double-encoded path directly in the URL string —
        do NOT use httpx params= which would triple-encode it.

  downloadBatch.do?course_id=...&gradeBookPK=...&isGroup=false
      → ZIP of all submitted files in the same order as getStudentWork.do.
        Used as a fast alternative to per-student fetching when no whitelist.

BB REST API is used to map student number → BB internal user ID
(needed for grade submission via PATCH gradebook/columns/{col}/users/{uid}).
"""
from __future__ import annotations

import io
import re
import zipfile
from urllib.parse import quote, unquote

import httpx

from models import Attachment, Submission

HW_BASE = "/webapps/bb-homeWorkCheck-BBLEARN/homeWorkCheck"
BB_API = "/learn/api/public/v1"

# getStudentWork.do has two submission link formats:
#   1. href="...CheckAloneWork.do?...userId=X&filePk=Y&...&attemptPk=Z">查看</a> (already graded)
#   2. onclick="checkWork('userId','filePk','attemptPk')">批改</a> (needs grading)
_STUDENT_PATTERN = re.compile(
    r'<a[^>]*CheckAloneWork\.do\?course_id=[^&]+&gradeBookPK=(\d+)'
    r'&userId=(\d+)&filePk=(\d+)&title=([^&"]+)&attemptPk=(\d+)[^>]*>([^<]+)</a>'
)
_STUDENT_ONCLICK_PATTERN = re.compile(
    r"""onclick=['"]\s*checkWork\(\s*['"](\d+)['"]\s*,\s*['"](\d+)['"]\s*,\s*['"](\d+)['"]\s*,\s*['"]\d*['"]\s*\)[^>]*>([^<]+)</a>"""
)
_NAME_PATTERN = re.compile(
    r'scope="row"[^>]*>\s*(\d{10})\s*</th>.*?table-data-cell-value">(.*?)</span>',
    re.DOTALL,
)
_FILE_PATH_PATTERN = re.compile(r"(?:var filePath\s*=\s*|path:\s*)'(/usr/local/blackboard[^']+)'")


class PKUHomeworkCrawler:
    def __init__(self, client: httpx.Client, course_id: str, whitelist: set[str]):
        self.client = client
        self.course_id = course_id
        self.whitelist = whitelist
        self._bb_user_map: dict[str, str] = {}  # student_number → bb_internal_user_id

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_assignments(self) -> list[dict]:
        """Return list of assignments: [{id, name, gradeBookPK}, ...]."""
        resp = self.client.get(
            f"{HW_BASE}/getHomeWorkList.do",
            params={"course_id": self.course_id},
        )
        resp.raise_for_status()
        return _parse_homework_list(resp.text)

    def fetch_submissions(self, grade_book_pk: str, title: str) -> list[Submission]:
        """
        Fetch student submissions for one assignment.

        Strategy:
        - Whitelist set → per-student: CheckWork.do + api/pdf.do (2 reqs per student)
        - No whitelist  → batch ZIP: downloadBatch.do (1 req for all files, fast)
        """
        self._ensure_bb_user_map()

        resp = self.client.get(
            f"{HW_BASE}/getStudentWork.do",
            params={
                "course_id": self.course_id,
                "gradeBookPK": grade_book_pk,
                "title": title,
                "showAll": "true",
            },
        )
        resp.raise_for_status()
        students = _parse_student_list(resp.text)

        if not students:
            return []

        if self.whitelist:
            return self._fetch_per_student(students, grade_book_pk, title)
        else:
            return self._fetch_batch_zip(students, grade_book_pk, title)

    # ------------------------------------------------------------------
    # Fetching strategies
    # ------------------------------------------------------------------

    def _fetch_per_student(
        self, students: list[dict], grade_book_pk: str, title: str
    ) -> list[Submission]:
        """Download individual files via CheckWork.do + api/pdf.do."""
        submissions: list[Submission] = []
        for student in students:
            student_id = student["userId"]
            if student_id not in self.whitelist:
                continue

            file_bytes, filename, content_type = self._download_student_file(
                grade_book_pk=grade_book_pk,
                title=title,
                user_id=student_id,
                file_pk=student["filePk"],
                attempt_pk=student["attemptPk"],
            )
            if file_bytes is None:
                continue

            submissions.append(Submission(
                student_id=student_id,
                student_name=student["name"],
                assignment_id=grade_book_pk,
                assignment_title=title,
                bb_user_id=self._bb_user_map.get(student_id, ""),
                attachments=[Attachment(filename=filename, content_type=content_type, data=file_bytes)],
                submitted_at=student.get("submitted_at", ""),
                already_graded=student.get("already_graded", False),
            ))
        return submissions

    def _fetch_batch_zip(
        self, students: list[dict], grade_book_pk: str, title: str
    ) -> list[Submission]:
        """Download all files at once via downloadBatch.do ZIP.

        Falls back to per-student fetching if batch download fails."""
        try:
            zip_resp = self.client.get(
                f"{HW_BASE}/downloadBatch.do",
                params={"course_id": self.course_id, "gradeBookPK": grade_book_pk, "isGroup": "false"},
            )
            zip_resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                zip_files = [(name, zf.read(name)) for name in zf.namelist()]

            submissions: list[Submission] = []
            for student, (filename, file_bytes) in zip(students, zip_files):
                student_id = student["userId"]
                submissions.append(Submission(
                    student_id=student_id,
                    student_name=student["name"],
                    assignment_id=grade_book_pk,
                    assignment_title=title,
                    bb_user_id=self._bb_user_map.get(student_id, ""),
                    attachments=[Attachment(
                        filename=filename,
                        content_type=_guess_mime(filename),
                        data=file_bytes,
                    )],
                    submitted_at=student.get("submitted_at", ""),
                    already_graded=student.get("already_graded", False),
                ))
            return submissions
        except (httpx.HTTPStatusError, zipfile.BadZipFile) as e:
            # Fall back to per-student fetching if batch download fails
            import sys
            print(f"  [yellow]Batch download failed (will fetch individually): {e}[/yellow]", file=sys.stderr)
            # Temporarily set whitelist to all students to trigger per-student fetch
            original_whitelist = self.whitelist
            self.whitelist = {s["userId"] for s in students}
            try:
                return self._fetch_per_student(students, grade_book_pk, title)
            finally:
                self.whitelist = original_whitelist

    def _download_student_file(
        self, grade_book_pk: str, title: str, user_id: str, file_pk: str, attempt_pk: str
    ) -> tuple[bytes | None, str, str]:
        """
        Fetch CheckWork.do to extract filePath, then download via api/pdf.do.
        Returns (file_bytes, filename, content_type) or (None, "", "") on failure.
        """
        resp = self.client.get(
            f"{HW_BASE}/CheckWork.do",
            params={
                "course_id": self.course_id,
                "gradeBookPK": grade_book_pk,
                "userId": user_id,
                "filePk": file_pk,
                "title": title,
                "attemptPk": attempt_pk,
            },
        )
        resp.raise_for_status()

        m = _FILE_PATH_PATTERN.search(resp.text)
        if not m:
            return None, "", ""

        file_path = m.group(1)
        filename = file_path.rsplit("/", 1)[-1]
        content_type = _guess_mime(filename)

        # Double-encode and pass directly in URL (params= would triple-encode)
        encoded = quote(quote(file_path, safe=""), safe="")
        file_resp = self.client.get(f"{HW_BASE}/api/pdf.do?path={encoded}")
        file_resp.raise_for_status()

        return file_resp.content, filename, content_type

    # ------------------------------------------------------------------
    # BB user map (for grade submission)
    # ------------------------------------------------------------------

    def _ensure_bb_user_map(self) -> None:
        if self._bb_user_map:
            return
        try:
            users = self._bb_paginate(
                f"{BB_API}/courses/{self.course_id}/users",
                params={"fields": "userId,user.userName"},
            )
            for u in users:
                bb_uid = u.get("userId", "")
                student_number = u.get("user", {}).get("userName", "")
                if bb_uid and student_number:
                    self._bb_user_map[student_number] = bb_uid
        except httpx.HTTPStatusError:
            pass  # non-fatal

    def _bb_paginate(self, path: str, params: dict | None = None) -> list[dict]:
        params = dict(params or {})
        params.setdefault("limit", "200")
        results: list[dict] = []
        url: str | None = path
        while url:
            resp = self.client.get(url, params=params if url == path else None)
            resp.raise_for_status()
            body = resp.json()
            results.extend(body.get("results", []))
            url = body.get("paging", {}).get("nextPage")
        return results


# ------------------------------------------------------------------
# HTML parsers
# ------------------------------------------------------------------

def _parse_homework_list(html: str) -> list[dict]:
    """Extract assignment list from getHomeWorkList.do HTML."""
    pattern = re.compile(
        r'getStudentWork\.do\?[^"\']*?title=([^&"\']+)[^"\']*?gradeBookPK=(\d+)|'
        r'getStudentWork\.do\?[^"\']*?gradeBookPK=(\d+)[^"\']*?title=([^&"\']+)'
    )
    seen: set[str] = set()
    assignments: list[dict] = []
    for m in pattern.finditer(html):
        title = unquote(m.group(1) or m.group(4) or "")
        pk = m.group(2) or m.group(3) or ""
        if pk and pk not in seen:
            seen.add(pk)
            assignments.append({"id": f"_{pk}_1", "name": title, "gradeBookPK": pk})
    return assignments


def _parse_student_list(html: str) -> list[dict]:
    """Extract submitted student list from getStudentWork.do HTML.

    Handles two link formats:
    - CheckAloneWork.do href with "查看" (view, already graded)
    - onclick="checkWork('userId','filePk','attemptPk')" with "批改" (grade, needs grading)

    For students with multiple attempts, keeps the newest one (highest attemptPk).
    Adds an 'already_graded' field to indicate if the attempt is already graded.
    """
    names = {m.group(1): m.group(2).strip() for m in _NAME_PATTERN.finditer(html)}
    student_map: dict[str, dict] = {}  # userId -> best attempt

    for m in _STUDENT_PATTERN.finditer(html):
        _, user_id, file_pk, title_enc, attempt_pk, link_text = m.groups()
        already_graded = link_text.strip() == "查看"
        attempt = {
            "userId": user_id,
            "filePk": file_pk,
            "attemptPk": attempt_pk,
            "name": names.get(user_id, "Unknown"),
            "already_graded": already_graded,
        }
        # Keep the one with higher attemptPk (newer attempt)
        if user_id not in student_map or int(attempt_pk) > int(student_map[user_id]["attemptPk"]):
            student_map[user_id] = attempt

    for m in _STUDENT_ONCLICK_PATTERN.finditer(html):
        user_id, file_pk, attempt_pk, link_text = m.groups()
        already_graded = link_text.strip() == "查看"
        attempt = {
            "userId": user_id,
            "filePk": file_pk,
            "attemptPk": attempt_pk,
            "name": names.get(user_id, "Unknown"),
            "already_graded": already_graded,
        }
        # Keep the one with higher attemptPk (newer attempt)
        if user_id not in student_map or int(attempt_pk) > int(student_map[user_id]["attemptPk"]):
            student_map[user_id] = attempt

    return list(student_map.values())


def _guess_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "zip": "application/zip",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
    }.get(ext, "application/octet-stream")
