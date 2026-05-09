# Tool Executor Server

FastAPI 기반 프록시 서버로, OpenWebUI LLM의 응답에서 자동으로 빌트인 도구를 실행하며 SSE(Streaming Server Events)를 통해 실시간 결과를 전달합니다.

## 개요

LLM이 `tool_calls`를 반환할 때, 서버가 자동으로 도구를 실행하고 결과를 다시 LLM에 전달하는 루프를 제공합니다. 이를 통해 함수 호출(Function Calling)을 손쉽게 구현할 수 있습니다.

### 동작 흐름

1. OpenAI 호환 형식의 `/chat/completions` 요청 수신
2. OpenWebUI로 스트리밍 요청 전달
3. LLM 토큰을 실시간으로 클라이언트에 전달
4. `tool_calls` 감지 시 도구 실행, 결과를 메시지에 추가하고 재요청
5. 종료 시 최종 SSE `[DONE]` 이벤트 전송

## 설치

```bash
# 개발 모드로 설치
pip install -e .
```

## 사용 방법

### 서버 관리 명령어

`tool-executer-server` 명령어를 통해 서버를 시작, 중지, 상태 확인할 수 있습니다.

```bash
# 서버 시작
tool-executer-server start

# 서버 상태 확인
tool-executer-server status

# 서버 중지
tool-executer-server stop
```

### API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/chat/completions` | POST | OpenAI 호환 채팅 완성 (스트리밍/비스트리밍 지원) |
| `/health` | GET | 서버 health 확인 |
| `/tools` | GET | 사용 가능한 도구 목록 조회 |

## 프로젝트 구조

```
tool_executor/
├── pyproject.toml              # 패키지 메타데이터 및 빌드 설정
├── requirements.txt            # 의존성 목록
├── .env.example                # 환경 변수 예시
├── README.md                   # 이 파일
└── src/
    └── tool_executor/
        ├── __init__.py          # 패키지 초기화
        ├── cli.py               # CLI 진입점 (start/stop/status)
        ├── main.py              # FastAPI 앱 및 엔드포인트
        ├── config.py            # 설정 및 환경 변수 관리
        ├── client.py            # OpenWebUI API 클라이언트
        ├── models.py            # Pydantic 데이터 모델
        ├── executor.py          # 코드 실행 엔진 (스레드 풀 기반)
        └── tools.py             # 빌트인 도구 구현 및 디스패치
```

## 환경 변수

`.env` 파일을 사용하거나 환경 변수를 직접 설정할 수 있습니다. `.env.example`을 참고하여 복사 후 수정하세요.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OPENWEBUI_BASE_URL` | `http://localhost:8080` | OpenWebUI API 주소 |
| `OPENWEBUI_API_KEY` | (없음) | OpenWebUI API 키 (필수) |
| `MAX_TOOL_ITERATIONS` | `10` | 도구 호출 최대 반복 횟수 |
| `CODE_EXEC_TIMEOUT` | `60` | 코드 실행 타임아웃(초) |
| `CODE_EXEC_MAX_WORKERS` | `5` | 코드 실행 스레드 풀 크기 |
| `SERVER_HOST` | `0.0.0.0` | 서버 바인딩 주소 |
| `SERVER_PORT` | `8000` | 서버 포트 |
| `OPENWEBUI_TIMEOUT` | `300` | OpenWebUI API 요청 타임아웃(초) |