# uv 사용법(poetry 와 비교해서)

## 🚀 uv 소개

uv는 Rust로 작성된 초고속 Python 패키지 관리자로, Poetry와 유사한 기능을 제공하면서도 10-100배 빠른 성능을 자랑합니다.

## 📦 설치

### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### pip을 통한 설치

```bash
pip install uv
```

---

## 🔄 Poetry vs uv 명령어 비교

| 작업        | Poetry                        | uv                              |
|-----------|-------------------------------|---------------------------------|
| 프로젝트 초기화  | `poetry init`                 | `uv init`                       |
| 의존성 설치    | `poetry install`              | `uv sync`                       |
| 패키지 추가    | `poetry add package`          | `uv add package`                |
| 개발 의존성 추가 | `poetry add --dev package`    | `uv add --dev package`          |
| 패키지 제거    | `poetry remove package`       | `uv remove package`             |
| 패키지 업데이트  | `poetry update`               | `uv lock --upgrade` → `uv sync` |
| 가상환경 활성화  | `poetry shell`                | `source .venv/bin/activate`     |
| 스크립트 실행   | `poetry run python script.py` | `uv run python script.py`       |
| 패키지 목록 확인 | `poetry show`                 | `uv pip list`                   |
| 빌드        | `poetry build`                | `uv build`                      |

---

## 🎯 주요 uv 명령어

### 1. 프로젝트 생성 및 초기화

```bash
# 새 프로젝트 생성
uv init my-project
cd my-project

# 기존 프로젝트에서 초기화 (pyproject.toml 생성)
uv init
```

### 2. Python 버전 관리

```bash
# Python 버전 설치
uv python install 3.12

# 프로젝트에 Python 버전 지정
uv python pin 3.12

# 사용 가능한 Python 버전 확인
uv python list
```

### 3. 의존성 관리

```bash
# 의존성 설치 (pyproject.toml 기반)
uv sync

# 패키지 추가
uv add requests
uv add "django>=4.0"

# 개발 의존성 추가
uv add --dev pytest black ruff

# 선택적 의존성 그룹 추가
uv add --optional-group docs sphinx

# 패키지 제거
uv remove requests

# lock 파일 업데이트
uv lock

# 모든 패키지 업데이트
uv lock --upgrade
uv sync
```

### 4. 가상환경 관리

```bash
# 가상환경 생성 (자동으로 .venv에 생성됨)
uv venv

# 가상환경 활성화
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 가상환경 없이 명령 실행
uv run python script.py
uv run pytest
```

### 5. 패키지 정보 확인

```bash
# 설치된 패키지 목록
uv pip list

# 패키지 트리 구조 확인
uv pip tree

# 패키지 정보 상세 확인
uv pip show package-name
```

---

## 📁 프로젝트 구조

### Poetry 프로젝트 구조

```
my-project/
├── pyproject.toml      # 프로젝트 설정
├── poetry.lock         # 잠금 파일
└── ...
```

### uv 프로젝트 구조

```
my-project/
├── pyproject.toml      # 프로젝트 설정 (Poetry와 호환)
├── uv.lock            # 잠금 파일
├── .venv/             # 가상환경 (자동 생성)
└── ...
```

---

## 🔧 pyproject.toml 예시

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My awesome project"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev-dependencies = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## ⚡ 자주 사용하는 워크플로우

### 일상적인 개발 작업

```bash
# 프로젝트 클론 후 설정
git clone <repo>
cd <repo>
uv sync                    # 의존성 설치

# 개발 시작
source .venv/bin/activate  # 가상환경 활성화
uv run python main.py      # 또는 직접 실행

# 새 패키지 추가 후 커밋
uv add pandas
git add pyproject.toml uv.lock
git commit -m "Add pandas dependency"

# 테스트 실행
uv run pytest
# 또는
uv run python -m pytest

# 코드 포맷팅
uv run black .
uv run ruff check .
```

---

## 📚 추가 리소스

- [uv 공식 문서](https://docs.astral.sh/uv/)
- [uv GitHub](https://github.com/astral-sh/uv)
