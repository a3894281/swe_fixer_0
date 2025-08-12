from difflib import SequenceMatcher
from coding.schemas import Patch, Edit


def create_patch(original_files: dict[str, str], edited_files: dict[str, str]) -> Patch:
    """
    Enhanced patch creation with better diff handling and comprehensive validation
    """
    if not edited_files:
        return Patch(edits=[])

    # Use optimized patch creation for better results
    optimized_patch = create_optimized_patch(original_files, edited_files)
    
    # Validate the patch before returning
    is_valid, validation_errors = validate_patch(optimized_patch, original_files)
    
    if not is_valid:
        print(f"⚠️ Patch validation issues: {validation_errors}")
        # Fall back to basic patch creation if optimized fails
        edits = []
        for filename in edited_files:
            if filename not in original_files:
                # Handle new files by creating edits for each line
                new_lines = edited_files[filename].splitlines()
                for i, line in enumerate(new_lines):
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=i,
                            line_content="",
                            new_line_content=line,
                        )
                    )
                continue

            original_content = original_files[filename]
            edited_content = edited_files[filename]

            # Skip if no changes
            if original_content == edited_content:
                continue

            # Create edits for this file
            file_edits = create_file_edits(filename, original_content, edited_content)
            edits.extend(file_edits)

        return Patch(edits=edits)
    
    return optimized_patch


def create_file_edits(
    filename: str, original_content: str, edited_content: str
) -> list[Edit]:
    """
    Create edit objects for a single file with improved accuracy
    """
    edits = []

    # Split into lines
    original_lines = original_content.splitlines(keepends=False)
    edited_lines = edited_content.splitlines(keepends=False)

    # Use SequenceMatcher to get detailed diff information
    matcher = SequenceMatcher(None, original_lines, edited_lines)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # No changes in this section
            continue
        elif tag == "replace":
            # Lines were modified
            # Handle replacement as delete old + insert new
            max_changes = max(i2 - i1, j2 - j1)
            for idx in range(max_changes):
                old_line = original_lines[i1 + idx] if (i1 + idx) < i2 else ""
                new_line = edited_lines[j1 + idx] if (j1 + idx) < j2 else ""

                # Only create edit if there's actually a change
                if old_line != new_line:
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=i1 + idx,
                            line_content=old_line,
                            new_line_content=new_line,
                        )
                    )
        elif tag == "delete":
            # Lines were deleted
            for idx in range(i1, i2):
                if idx < len(original_lines):
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=idx,
                            line_content=original_lines[idx],
                            new_line_content="",
                        )
                    )
        elif tag == "insert":
            # Lines were inserted
            for idx in range(j1, j2):
                if idx < len(edited_lines):
                    # Insert at the position where deletion occurred, or at the end
                    insert_position = i1 + (idx - j1)
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=insert_position,
                            line_content="",
                            new_line_content=edited_lines[idx],
                        )
                    )

    return edits


def create_optimized_patch(
    original_files: dict[str, str], edited_files: dict[str, str]
) -> Patch:
    """
    Create an optimized patch that minimizes the number of edits while maintaining accuracy
    """
    if not edited_files:
        return Patch(edits=[])

    all_edits = []

    for filename in edited_files:
        if filename not in original_files:
            continue

        original_content = original_files[filename]
        edited_content = edited_files[filename]

        if original_content == edited_content:
            continue

        # Get optimized edits for this file
        file_edits = create_optimized_file_edits(
            filename, original_content, edited_content
        )
        all_edits.extend(file_edits)

    return Patch(edits=all_edits)


def create_optimized_file_edits(
    filename: str, original_content: str, edited_content: str
) -> list[Edit]:
    """
    Create optimized edits by grouping related changes and minimizing edit count
    """
    original_lines = original_content.splitlines(keepends=False)
    edited_lines = edited_content.splitlines(keepends=False)

    # Use a more sophisticated diff approach
    differ = SequenceMatcher(None, original_lines, edited_lines)
    edits = []

    # Get grouped operations
    grouped_ops = []
    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag != "equal":
            grouped_ops.append((tag, i1, i2, j1, j2))

    # Process grouped operations
    for tag, i1, i2, j1, j2 in grouped_ops:
        if tag == "replace":
            # Handle replacements more efficiently
            edits.extend(
                handle_replacement(
                    filename, original_lines, edited_lines, i1, i2, j1, j2
                )
            )
        elif tag == "delete":
            # Handle deletions
            for idx in range(i1, i2):
                if idx < len(original_lines):
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=idx,
                            line_content=original_lines[idx],
                            new_line_content="",
                        )
                    )
        elif tag == "insert":
            # Handle insertions
            for idx in range(j1, j2):
                if idx < len(edited_lines):
                    # Insert after the last unchanged line before this insertion
                    insert_position = i1 + (idx - j1)
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=insert_position,
                            line_content="",
                            new_line_content=edited_lines[idx],
                        )
                    )

    return edits


def handle_replacement(
    filename: str,
    original_lines: list[str],
    edited_lines: list[str],
    i1: int,
    i2: int,
    j1: int,
    j2: int,
) -> list[Edit]:
    """
    Handle replacement operations more efficiently
    """
    edits = []

    # Get the original and new sections
    original_section = original_lines[i1:i2]
    edited_section = edited_lines[j1:j2]

    # If the sections are the same length, do line-by-line replacement
    if len(original_section) == len(edited_section):
        for idx, (old_line, new_line) in enumerate(
            zip(original_section, edited_section)
        ):
            if old_line != new_line:
                edits.append(
                    Edit(
                        file_name=filename,
                        line_number=i1 + idx,
                        line_content=old_line,
                        new_line_content=new_line,
                    )
                )
    else:
        # Handle different lengths by treating as delete + insert
        # Delete original lines
        for idx, line in enumerate(original_section):
            edits.append(
                Edit(
                    file_name=filename,
                    line_number=i1 + idx,
                    line_content=line,
                    new_line_content="",
                )
            )

        # Insert new lines
        for idx, line in enumerate(edited_section):
            edits.append(
                Edit(
                    file_name=filename,
                    line_number=i1 + idx,
                    line_content="",
                    new_line_content=line,
                )
            )

    return edits


def validate_patch(
    patch: Patch, original_files: dict[str, str]
) -> tuple[bool, list[str]]:
    """
    Validate that a patch makes sense and can be applied
    """
    issues = []

    if not patch.edits:
        issues.append("Patch contains no edits")
        return False, issues

    # Group edits by file
    files_with_edits = {}
    for edit in patch.edits:
        if edit.file_name not in files_with_edits:
            files_with_edits[edit.file_name] = []
        files_with_edits[edit.file_name].append(edit)

    # Validate each file's edits
    for filename, file_edits in files_with_edits.items():
        if filename not in original_files:
            issues.append(f"File {filename} not found in original files")
            continue

        original_lines = original_files[filename].splitlines()

        # Check line number validity
        for edit in file_edits:
            if edit.line_number < 0:
                issues.append(
                    f"Invalid negative line number {edit.line_number} in {filename}"
                )
            elif edit.line_number >= len(original_lines) and edit.line_content:
                issues.append(
                    f"Line number {edit.line_number} exceeds file length in {filename}"
                )

    # Check for conflicting edits (multiple edits on same line)
    line_edit_count = {}
    for edit in patch.edits:
        key = (edit.file_name, edit.line_number)
        if key in line_edit_count:
            line_edit_count[key] += 1
        else:
            line_edit_count[key] = 1

    conflicting_lines = [key for key, count in line_edit_count.items() if count > 1]
    if conflicting_lines:
        issues.append(f"Multiple edits on same lines: {conflicting_lines}")

    return len(issues) == 0, issues


def apply_patch_preview(original_files: dict[str, str], patch: Patch) -> dict[str, str]:
    """
    Preview what files would look like after applying the patch (for validation)
    """
    if not patch.edits:
        return original_files.copy()

    result_files = {}

    # Group edits by file
    files_with_edits = {}
    for edit in patch.edits:
        if edit.file_name not in files_with_edits:
            files_with_edits[edit.file_name] = []
        files_with_edits[edit.file_name].append(edit)

    # Apply edits to each file
    for filename, file_edits in files_with_edits.items():
        if filename not in original_files:
            # New file
            lines = []
            for edit in sorted(file_edits, key=lambda e: e.line_number):
                if edit.new_line_content:
                    lines.append(edit.new_line_content)
            result_files[filename] = "\n".join(lines)
        else:
            # Existing file - apply edits
            original_lines = original_files[filename].splitlines()
            result_lines = original_lines.copy()

            # Sort edits by line number (descending to avoid index shifting issues)
            sorted_edits = sorted(file_edits, key=lambda e: e.line_number, reverse=True)

            for edit in sorted_edits:
                line_num = edit.line_number

                if line_num < len(result_lines):
                    if edit.new_line_content == "":
                        # Delete line
                        if line_num < len(result_lines):
                            del result_lines[line_num]
                    else:
                        # Replace line
                        result_lines[line_num] = edit.new_line_content
                else:
                    # Insert new line
                    result_lines.append(edit.new_line_content)

            result_files[filename] = "\n".join(result_lines)

    # Include unchanged files
    for filename, content in original_files.items():
        if filename not in result_files:
            result_files[filename] = content

    return result_files


def get_patch_statistics(patch: Patch) -> dict[str, int]:
    """
    Get statistics about the patch for analysis
    """
    stats = {
        "total_edits": len(patch.edits),
        "files_modified": len(set(edit.file_name for edit in patch.edits)),
        "lines_added": 0,
        "lines_deleted": 0,
        "lines_modified": 0,
    }

    for edit in patch.edits:
        if edit.line_content == "" and edit.new_line_content != "":
            stats["lines_added"] += 1
        elif edit.line_content != "" and edit.new_line_content == "":
            stats["lines_deleted"] += 1
        elif edit.line_content != "" and edit.new_line_content != "":
            stats["lines_modified"] += 1

    return stats
