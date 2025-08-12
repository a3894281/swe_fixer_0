import os


def load_directory(directory: str) -> dict[str, str]:
    """
    Enhanced directory loader with better file handling and content analysis
    """
    repo_files = {}

    # Skip these directories and files
    skip_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".tox",
        "dist",
        "build",
        ".mypy_cache",
        "coverage",
        ".coverage",
        "htmlcov",
        ".nox",
    }
    skip_extensions = {
        ".pyc",
        ".pyo",
        ".log",
        ".tmp",
        ".cache",
        ".lock",
        ".pid",
        ".swp",
        ".bak",
        ".orig",
    }

    for root, dirs, files in os.walk(directory):
        # Remove skip directories from dirs list
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        # Get relative path from repo root
        rel_path = os.path.relpath(root, directory)

        for filename in files:
            # Skip unwanted file extensions
            if any(filename.endswith(ext) for ext in skip_extensions):
                continue

            # Skip hidden files (except important ones)
            if filename.startswith(".") and filename not in {
                ".env",
                ".env.example",
                ".gitignore",
                ".dockerignore",
            }:
                continue

            file_path = os.path.join(root, filename)

            # Get the relative path for the repo_files dict key
            if rel_path == ".":
                repo_key = filename
            else:
                repo_key = os.path.join(rel_path, filename)

            try:
                # Try to read file with different encodings
                content = read_file_safely(file_path)
                if content is not None:
                    repo_files[repo_key] = content
            except Exception as e:
                print(f"Warning: Could not read file {repo_key}: {e}")
                continue

    return repo_files


def read_file_safely(file_path: str) -> str | None:
    """
    Safely read a file with multiple encoding attempts
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    # Skip binary files by checking file extension
    binary_extensions = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".wav",
        ".ogg",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
        ".bz2",
        ".xz",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    }

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in binary_extensions:
        return None

    # Check if file is too large (skip files larger than 2MB for better coverage)
    try:
        if os.path.getsize(file_path) > 2 * 1024 * 1024:  # 2MB limit
            print(f"Skipping large file: {file_path}")
            return None
    except OSError:
        return None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                content = f.read()

                # Basic check if this is likely a text file
                if is_likely_text_content(content):
                    return content
                else:
                    return None

        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            return None

    return None


def is_likely_text_content(content: str, max_check_length: int = 1000) -> bool:
    """
    Check if content is likely to be text (not binary)
    """
    if not content:
        return True

    # Check first part of content
    check_content = content[:max_check_length]

    # Count non-printable characters
    non_printable = sum(1 for c in check_content if ord(c) < 32 and c not in "\n\r\t")

    # If more than 10% non-printable characters, likely binary
    if len(check_content) > 0 and non_printable / len(check_content) > 0.1:
        return False

    # Check for null bytes (strong indicator of binary content)
    if "\x00" in check_content:
        return False

    return True
