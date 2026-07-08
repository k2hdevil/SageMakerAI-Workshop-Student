# SageMaker AI 워크샵 — 학생 중도탈락 예측 (Track A: 빌트인 XGBoost)

학생의 인적사항·성적·거시경제 지표로 **졸업 / 중도탈락 / 재학**을 예측하는 다중 클래스 분류
모델을, Amazon SageMaker로 **전처리 → 학습 → 평가 → 튜닝 → 배포 → 추론**까지 직접 만들어 보는
실습형 워크샵입니다. 마지막에는 **선택 과제**로 엔드포인트를 호출하는 웹앱을 로컬 서버로 띄웁니다.

참가자는 노트북 곳곳의 `____` 빈칸과 `# TODO` 를 힌트/문서 링크를 참고해 직접 채웁니다.

## 대상 참가자
- AWS 기본 서비스(S3, IAM)를 사용해 본 적 있는 **초중급 ML 엔지니어 / 데이터 사이언티스트**
- Python, pandas, scikit-learn 기초 사용 가능
- SageMaker를 처음 접하거나, SDK 기반 워크플로우를 체계적으로 배우고 싶은 분

## 학습 목표
워크샵을 마치면 참가자는 다음을 할 수 있습니다:
1. SageMaker **Processing Job**으로 데이터 전처리 파이프라인을 구성할 수 있다
2. **빌트인 XGBoost 알고리즘**으로 학습 작업을 실행하고 CloudWatch 로그에서 학습 곡선을 확인할 수 있다
3. **Batch Transform**으로 대량 오프라인 추론을 수행하고 confusion matrix / macro-F1 등으로 모델을 평가할 수 있다
4. **Automatic Model Tuning(HPO)**으로 하이퍼파라미터를 베이지안 최적화할 수 있다
5. 학습된 모델을 **실시간 엔드포인트**로 배포하고 boto3/SDK로 호출할 수 있다
6. 배포된 엔드포인트를 **Gradio 웹앱**으로 감싸 데모를 만들 수 있다
7. 워크샵 종료 후 모든 리소스를 **체계적으로 정리**할 수 있다

## 소요 시간

총 **약 4시간** (선택 과제 포함 4시간 30분). 학습/배치/배포 작업 대기 시간과 수강생의 코드 작성 시간이 포함됩니다.

## 워크플로우 아키텍처
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 01. 전처리   │────▶│ 02. 학습     │────▶│ 03. 평가     │────▶│ 04. 튜닝     │
│ Processing  │     │ Training    │     │ Batch       │     │ HPO         │
│ Job         │     │ Job         │     │ Transform   │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                    ┌─────────────┐     ┌─────────────┐            │
                    │ 06. 추론     │◀────│ 05. 배포     │◀───────────┘
                    │ SDK/boto3   │     │ Endpoint    │
                    └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐     ┌─────────────┐
                    │ 07. 웹앱    │     │ 08. 정리     │
                    │ Gradio      │     │ Cleanup     │
                    └─────────────┘     └─────────────┘
```

## 데이터셋
- `dataset/data.csv` — UCI *Predict Students' Dropout and Academic Success* (세미콜론 구분)
  - 출처: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
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
├── webapp/                   # (사용 안 함 - Gradio는 노트북 내 실행)
├── _build/                   # 노트북 생성기(유지보수용) — 실습에는 불필요
└── README.md
```

## 실행 방법 (SageMaker Studio)
1. 이 폴더를 Studio에 올리거나 git으로 클론합니다.
2. `notebooks/` 폴더의 노트북을 **번호 순서대로** 엽니다.
3. 커널을 SageMaker SDK 포함 Python 3 이미지로 선택합니다.
4. 위에서 아래로 셀을 실행하며 `____` / `# TODO` 를 채웁니다.
5. 노트북 간 값은 `%store` 로 자동 전달되므로 **순서대로** 실행해야 합니다.

> 노트북 작업 디렉토리는 `notebooks/`이므로 데이터 상대경로는 `../dataset/data.csv` 입니다.

## 노트북 순서
| # | 노트북 | 단계 | SageMaker 기능 | 소요 시간 |
|---|--------|------|----------------|-----------|
| 01 | `01_preprocessing.ipynb` | 환경설정·EDA·전처리 | Processing Job (SKLearnProcessor) | 40분 |
| 02 | `02_training.ipynb` | 모델 학습 | Training Job (빌트인 XGBoost) | 30분 |
| 03 | `03_evaluation.ipynb` | 평가 | Batch Transform + 지표 | 30분 |
| 04 | `04_tuning.ipynb` | 튜닝 | Automatic Model Tuning (HPO) | 50분 |
| 05 | `05_deployment.ipynb` | 배포 | 실시간 Endpoint | 20분 |
| 06 | `06_inference.ipynb` | 추론 | Endpoint 호출 (SDK/boto3) | 20분 |
| 07 | `07_web_app.ipynb` | (선택) 웹앱 | Gradio (노트북 내 실행, share=True) | 30분 |
| 08 | `08_cleanup.ipynb` | 리소스 정리 | Endpoint·Model·S3 전체 삭제 | 10분 |

난이도는 뒤로 갈수록 스캐폴딩을 줄였습니다. `06`, `07` 은 힌트만 주는 **개방형**입니다.

## ⚠️ 비용 & 리소스 정리
- **실시간 엔드포인트는 삭제 전까지 시간당 과금**됩니다.
- `06` 또는 `07` 의 **정리(cleanup) 셀**을 반드시 실행해 Endpoint / Endpoint config / Model 을 삭제하세요.
- 실습 후 SageMaker 콘솔의 **Endpoints / Models / Endpoint configurations** 에 남은 리소스가 없는지 확인하세요.

## 선택 과제: 웹 애플리케이션 (Gradio)
`07_web_app.ipynb` 에서 **Gradio**로 웹 UI를 만들어 노트북 셀에서 바로 실행합니다.
`share=True` 옵션으로 공개 URL이 자동 생성되어 별도 포트/프록시/방화벽 설정 없이 브라우저로 접속할 수 있습니다.

![Gradio 웹앱 스크린샷](assets/gradio_app_screenshot.png)

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


---

## 제작 정보
본 워크샵은 [Kiro](https://kiro.dev) 👻로 생성하였으며, HITL을 통해서 컨텐츠의 정확성을 검수했습니다.
