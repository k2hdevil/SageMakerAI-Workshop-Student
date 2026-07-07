# SageMaker AI 워크샵 — 학생 중도탈락 예측 (Track A: 빌트인 XGBoost)

학생의 인적사항·성적·거시경제 지표로 **졸업 / 중도탈락 / 재학**을 예측하는 다중 클래스 분류
모델을, Amazon SageMaker로 **전처리 → 학습 → 평가 → 튜닝 → 배포 → 추론**까지 직접 만들어 보는
실습형 워크샵입니다. 마지막에는 **선택 과제**로 엔드포인트를 호출하는 웹앱을 로컬 서버로 띄웁니다.

참가자는 노트북 곳곳의 `____` 빈칸과 `# TODO` 를 힌트/문서 링크를 참고해 직접 채웁니다.
막히면 강사용 정답(`solutions/`)의 같은 셀을 참고하세요.

## 데이터셋
- `dataset/data.csv` — UCI *Predict Students' Dropout and Academic Success* (세미콜론 구분)
- 4,424행 × 37열 (36 피처 + `Target`)
- `Target`: `Graduate`(≈50%) / `Dropout`(≈32%) / `Enrolled`(≈18%) — **클래스 불균형** 존재

## 사전 준비
1. **SageMaker Studio** 도메인과 사용자 프로필 (JupyterLab 스페이스)
2. 노트북에 연결된 **실행 역할(Execution Role)** — S3 접근 + SageMaker(학습/처리/엔드포인트) 권한
   (워크샵/실습 계정이라면 `AmazonSageMakerFullAccess` 로 시작해도 됩니다)
3. **SageMaker Python SDK 가 포함된 Python 3 커널** (예: *Data Science* / *Base Python* 이미지)
4. 서비스 쿼터: `ml.m5.xlarge`(학습·처리·배치), `ml.m5.large`(실시간 엔드포인트) 사용 가능해야 함

## 리포지토리 구조
```
SageMakerWorkshop/
├── dataset/data.csv          # 원본 데이터
├── notebooks/                # 참가자용 (빈칸형)   ← 여기서 실습
├── solutions/                # 강사용 (정답 완성본)
├── webapp/                   # 07 실행 시 생성됨 (Streamlit 앱)
├── _build/                   # 노트북 생성기(유지보수용) — 실습에는 불필요
└── README.md
```

## 실행 방법 (SageMaker Studio)
1. 이 폴더를 Studio에 올리거나 git으로 클론합니다.
2. `notebooks/` 폴더의 노트북을 **번호 순서대로** 엽니다.
3. 커널을 SageMaker SDK 포함 Python 3 이미지로 선택합니다.
4. 위에서 아래로 셀을 실행하며 `____` / `# TODO` 를 채웁니다.
5. 노트북 간 값은 `%store` 로 자동 전달되므로 **순서대로** 실행해야 합니다.

> 노트북 작업 디렉토리는 `notebooks/`(또는 `solutions/`) 이므로 데이터 상대경로는 `../dataset/data.csv` 입니다.

## 노트북 순서
| # | 노트북 | 단계 | SageMaker 기능 |
|---|--------|------|----------------|
| 01 | `01_preprocessing.ipynb` | 환경설정·EDA·전처리 | Processing Job (SKLearnProcessor) |
| 02 | `02_training.ipynb` | 모델 학습 | Training Job (빌트인 XGBoost) |
| 03 | `03_evaluation.ipynb` | 평가 | Batch Transform + 지표 |
| 04 | `04_tuning.ipynb` | 튜닝 | Automatic Model Tuning (HPO) |
| 05 | `05_deployment.ipynb` | 배포 | 실시간 Endpoint |
| 06 | `06_inference.ipynb` | 추론·정리 | Endpoint 호출 (SDK/boto3) + Cleanup |
| 07 | `07_web_app.ipynb` | (선택) 웹앱 | Streamlit 로컬 서버 → Endpoint |

난이도는 뒤로 갈수록 스캐폴딩을 줄였습니다. `06`, `07` 은 힌트만 주는 **개방형**입니다.

## ⚠️ 비용 & 리소스 정리
- **실시간 엔드포인트는 삭제 전까지 시간당 과금**됩니다.
- `06` 또는 `07` 의 **정리(cleanup) 셀**을 반드시 실행해 Endpoint / Endpoint config / Model 을 삭제하세요.
- 실습 후 SageMaker 콘솔의 **Endpoints / Models / Endpoint configurations** 에 남은 리소스가 없는지 확인하세요.

## 선택 과제: 웹 애플리케이션 (로컬 개발 서버)
`07_web_app.ipynb` 가 `webapp/app.py`(Streamlit)와 `requirements.txt` 를 생성합니다.
Streamlit 서버는 **터미널**에서 실행합니다(노트북 셀 아님):
```bash
cd webapp
pip install -r requirements.txt
ENDPOINT_NAME=student-success-endpoint AWS_REGION=<your-region> streamlit run app.py --server.port 8501
```
- **Studio**: `File ▸ New ▸ Terminal` 로 실행 후, 노트북 URL 경로를 `/proxy/8501/` 로 바꿔 접속
- **로컬 PC**: `aws configure` 로 자격증명 설정 후 `http://localhost:8501` 접속
  (IAM 권한에 `sagemaker:InvokeEndpoint` 필요)

## 노트북 재생성 (유지보수용)
참가자용/정답용 노트북은 단일 소스 `_build/gen.py` 에서 생성됩니다. 내용을 수정하려면
`_build/gen.py` 를 편집한 뒤 다시 생성하세요:
```bash
python3 _build/gen.py       # notebooks/ 와 solutions/ 재생성
python3 _build/validate.py  # 문법/빈칸 검증
```

## 트러블슈팅
- **`%store` 값이 없다고 나올 때**: 이전 번호 노트북을 먼저 실행했는지 확인하세요.
- **`AccessDenied` (S3/SageMaker)**: 실행 역할 권한을 확인하세요.
- **엔드포인트 호출 실패**: Endpoint가 `InService` 인지, 리전/자격증명이 맞는지 확인하세요.
- **입력 피처 순서 오류**: 빌트인 XGBoost는 학습 시와 **동일한 열 순서·헤더 없음·라벨 맨 앞** 규칙을 지켜야 합니다.
