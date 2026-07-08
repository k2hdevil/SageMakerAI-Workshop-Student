#!/usr/bin/env python3
"""Generate the SageMaker Studio workshop notebooks.

For every notebook we define its cells ONCE. Code cells can carry two variants:
a participant variant (with blanks / TODOs) and a solution variant (complete).
The generator emits:
  - notebooks/<name>.ipynb   -> participant variant (fill-in-the-blank)
  - solutions/<name>.ipynb   -> instructor variant (complete answers)

Run:  python3 _build/gen.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTICIPANT_DIR = os.path.join(ROOT, "notebooks")
SOLUTION_DIR = os.path.join(ROOT, "solutions")

# Portable kernel metadata. In SageMaker Studio, pick a Python 3 kernel backed by
# an image that ships the SageMaker Python SDK (e.g. "Data Science" / "Base Python").
KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "name": "python",
    "pygments_lexer": "ipython3",
    "codemirror_mode": {"name": "ipython", "version": 3},
}


# --- cell spec helpers -------------------------------------------------------
def md(text):
    """A markdown cell (identical in both variants)."""
    return ("md", text, text)


def code(src):
    """A code cell that is identical in both variants (boilerplate)."""
    return ("code", src, src)


def code2(participant, solution):
    """A code cell with a blanked participant variant and a complete solution."""
    return ("code", participant, solution)


NOTEBOOKS = []  # list of (filename, title, [cells])


def add(filename, title, cells):
    NOTEBOOKS.append((filename, title, cells))


# --- build -------------------------------------------------------------------
def _make_nb(cells, variant):
    nb = new_notebook()
    nb.metadata["kernelspec"] = dict(KERNELSPEC)
    nb.metadata["language_info"] = dict(LANGUAGE_INFO)
    for kind, participant_src, solution_src in cells:
        if kind == "md":
            nb.cells.append(new_markdown_cell(participant_src))
        else:
            src = participant_src if variant == "participant" else solution_src
            nb.cells.append(new_code_cell(src))
    return nb


def build():
    os.makedirs(PARTICIPANT_DIR, exist_ok=True)
    os.makedirs(SOLUTION_DIR, exist_ok=True)
    count = 0
    for filename, title, cells in NOTEBOOKS:
        for variant, folder in (("participant", PARTICIPANT_DIR), ("solution", SOLUTION_DIR)):
            nb = _make_nb(cells, variant)
            path = os.path.join(folder, filename)
            with open(path, "w", encoding="utf-8") as f:
                nbf.write(nb, f)
            # validate immediately
            nbf.read(path, as_version=4)
            count += 1
    print(f"OK: wrote and validated {count} notebooks "
          f"({len(NOTEBOOKS)} notebooks x 2 variants).")
    for filename, title, cells in NOTEBOOKS:
        print(f"  - {filename}: {len(cells)} cells  ({title})")


# =============================================================================
# Notebook 01 - Setup + EDA + Preprocessing (SageMaker Processing)
# =============================================================================
add("01_preprocessing.ipynb", "setup + EDA + preprocessing", [
    md(r"""# 01. 환경 설정 · 데이터 탐색 · 전처리

> **SageMaker AI 워크샵 (Track A: 빌트인 XGBoost)** — 학생 중도탈락 예측

이 워크샵에서는 학생의 인적사항·성적·거시경제 지표로 **졸업 / 중도탈락 / 재학**을 예측하는
다중 클래스 분류 모델을 SageMaker로 처음부터 끝까지 만들어 봅니다.

## 이 노트북에서 배우는 것
- SageMaker 세션 · 실행 역할(Execution Role) · 기본 S3 버킷 설정
- 원본 데이터를 S3에 업로드
- 간단한 탐색적 데이터 분석(EDA)과 **클래스 불균형** 확인
- **SageMaker Processing Job**으로 전처리(스케일링 없이 라벨 인코딩 + 분할) 수행

## 진행 방식
곳곳에 `____` 로 표시된 **빈칸**과 `# TODO` 주석이 있습니다. 힌트와 공식 문서 링크를 참고해
직접 채워 보세요.

> 이 노트북은 **SageMaker Studio JupyterLab**의 Python 3 커널(SageMaker Python SDK 포함
> 이미지, 예: *Data Science* / *Base Python*)에서 실행하는 것을 전제로 합니다.
"""),

    md(r"""## 0. 환경 설정

> 아래 셀은 SageMaker Python SDK **v2** 를 설치(또는 고정)합니다. 이 워크샵의 코드는 v2 API
> (`Estimator`, `HyperparameterTuner`, `Transformer` 등)에 맞춰져 있습니다.
>
> ⚠️ **설치 후 커널이 자동 재시작될 수 있습니다.** 재시작되면 아래 두 번째 셀부터 다시 실행하세요.
> (이 셀은 다시 실행할 필요 없습니다.)"""),
    code(r'''# SageMaker Python SDK v2 설치 (이미 v2라면 무시됨, v3가 설치되어 있으면 다운그레이드)
# Studio에서 커널의 작업 디렉토리가 노트북 위치와 다를 수 있어 먼저 맞춥니다.
import os, subprocess, sys, glob
try:
    os.getcwd()
except OSError:
    os.chdir(os.path.expanduser("~"))

# 노트북이 notebooks/ 또는 solutions/ 에 있을 때 ../dataset/data.csv 가 보이도록 cwd를 조정합니다.
if not os.path.exists("../dataset/data.csv"):
    candidates = glob.glob(os.path.expanduser("~/*/notebooks")) + \
                 glob.glob(os.path.expanduser("~/*/solutions"))
    for d in candidates:
        if os.path.exists(os.path.join(d, "../dataset/data.csv")):
            os.chdir(d)
            break

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker>=2,<3", "-qU"],
                      stdout=subprocess.DEVNULL)
import sagemaker
print(f"SageMaker SDK v{sagemaker.__version__} ready")
print(f"working dir: {os.getcwd()}")'''),
    code2(
        r'''import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
region = sess.boto_region_name

# TODO: SageMaker 실행 역할(IAM Role)과 기본 버킷을 가져오세요.
#  - 힌트: get_execution_role() 은 현재 노트북에 연결된 IAM 역할을 반환합니다.
#  - 힌트: sess.default_bucket() 은 계정/리전별 기본 버킷 이름을 반환(없으면 생성)합니다.
#  - 참고: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html
role = ____
bucket = ____

prefix = "sm-workshop-students"
print("region :", region)
print("role   :", role)
print("bucket :", bucket)
print("prefix :", prefix)''',
        r'''import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
region = sess.boto_region_name

role = get_execution_role()
bucket = sess.default_bucket()

prefix = "sm-workshop-students"
print("region :", region)
print("role   :", role)
print("bucket :", bucket)
print("prefix :", prefix)'''),

    md("## 1. 원본 데이터 S3 업로드\n\n리포지토리 루트의 `dataset/data.csv` 를 S3에 올립니다. (노트북 작업 디렉토리는 `notebooks/` 이므로 상대경로는 `../dataset/...` 입니다.)"),
    code2(
        r'''# TODO: 로컬 데이터셋을 S3에 업로드하고, 반환되는 S3 URI를 raw_s3 에 저장하세요.
#  - 힌트: sess.upload_data(path=..., bucket=..., key_prefix=...) 는 업로드된 S3 URI를 돌려줍니다.
#  - key_prefix 는 f"{prefix}/raw" 를 사용하세요.
#  - 참고: https://sagemaker.readthedocs.io/en/v2.219.0/api/utility/session.html
raw_s3 = ____
print("raw data:", raw_s3)''',
        r'''raw_s3 = sess.upload_data(path="../dataset/data.csv", bucket=bucket, key_prefix=f"{prefix}/raw")
print("raw data:", raw_s3)'''),

    md("## 2. 탐색적 데이터 분석 (EDA)"),
    code2(
        r'''import pandas as pd

# TODO: data.csv 는 세미콜론(;)으로 구분되어 있습니다. 올바른 구분자로 읽으세요.
#  - 참고: pandas.read_csv 의 sep 인자
df = pd.read_csv("../dataset/data.csv", sep="____")

# 컬럼명 끝에 섞여 있는 공백/탭 문자를 정리합니다(실무 데이터는 지저분합니다!).
df.columns = [c.strip() for c in df.columns]
print("shape:", df.shape)
df.head()''',
        r'''import pandas as pd

df = pd.read_csv("../dataset/data.csv", sep=";")

# 컬럼명 끝에 섞여 있는 공백/탭 문자를 정리합니다(실무 데이터는 지저분합니다!).
df.columns = [c.strip() for c in df.columns]
print("shape:", df.shape)
df.head()'''),

    code(r'''# 타깃 분포 확인
counts = df["Target"].value_counts()
ratio = df["Target"].value_counts(normalize=True).round(3)
summary = pd.concat([counts, ratio], axis=1)
summary.columns = ["count", "ratio"]
print(summary)

ax = counts.plot(kind="bar", title="Target distribution", rot=0)
ax.set_xlabel("class"); ax.set_ylabel("count")'''),

    md(r"""### 관찰
- 타깃은 **3개 클래스**: `Graduate`(≈50%), `Dropout`(≈32%), `Enrolled`(≈18%) — **불균형**이 있습니다.
- 따라서 정확도(accuracy)만으로 성능을 판단하면 소수 클래스(`Enrolled`)를 놓치기 쉽습니다.
  뒤의 평가 노트북에서 **macro-F1**과 **클래스별 지표**를 함께 보는 이유입니다.
- 모든 피처가 이미 수치형이라 별도 인코딩/스케일링 없이도 트리 기반 모델(XGBoost)에 바로 넣을 수 있습니다.
  전처리에서는 **타깃 라벨 인코딩**과 **학습/검증/테스트 분할**에 집중합니다."""),

    md(r"""## 3. 전처리 스크립트 작성

SageMaker Processing Job은 **별도 컨테이너**에서 스크립트를 실행합니다. 아래 셀은 `%%writefile`
매직으로 `src/preprocessing.py` 파일을 생성합니다.

> **빌트인 XGBoost 입력 규칙**: 학습/검증 CSV는 **헤더가 없어야 하며, 첫 번째 열이 정답(label)** 이어야 합니다.
> 이 규칙을 스크립트에서 지키는 것이 핵심 포인트입니다."""),
    code(r'''import os
os.makedirs("src", exist_ok=True)'''),
    code2(
        r'''%%writefile src/preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "/opt/ml/processing/input/data.csv"
LABEL_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}


def main():
    # TODO: 세미콜론 구분자로 읽으세요.
    df = pd.read_csv(INPUT, sep="____")
    df.columns = [c.strip() for c in df.columns]

    # TODO: 문자열 타깃을 정수로 인코딩하세요. (힌트: .str.strip().map(LABEL_MAP))
    df["Target"] = ____

    y = df["Target"]
    X = df.drop(columns=["Target"])

    # TODO: 클래스 비율을 유지하도록 stratify 옵션을 지정하세요. (70/15/15 분할)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=____)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=____)

    for name in ["train", "validation", "test"]:
        os.makedirs(f"/opt/ml/processing/{name}", exist_ok=True)

    # 빌트인 XGBoost용: label을 맨 앞 열로, 헤더 없이 저장
    train = pd.concat([y_train, X_train], axis=1)
    val = pd.concat([y_val, X_val], axis=1)
    # TODO: 학습/검증 CSV를 header 없이, index 없이 저장하세요.
    train.to_csv("/opt/ml/processing/train/train.csv", header=____, index=False)
    val.to_csv("/opt/ml/processing/validation/validation.csv", header=____, index=False)

    # 평가(배치 변환)용: 피처만(라벨 제외) / 정답 라벨은 따로 저장
    X_test.to_csv("/opt/ml/processing/test/test_x.csv", header=False, index=False)
    y_test.to_csv("/opt/ml/processing/test/test_y.csv", header=False, index=False)
    print("preprocessing done:", train.shape, val.shape, X_test.shape)


if __name__ == "__main__":
    main()''',
        r'''%%writefile src/preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "/opt/ml/processing/input/data.csv"
LABEL_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}


def main():
    df = pd.read_csv(INPUT, sep=";")
    df.columns = [c.strip() for c in df.columns]

    df["Target"] = df["Target"].str.strip().map(LABEL_MAP)

    y = df["Target"]
    X = df.drop(columns=["Target"])

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    for name in ["train", "validation", "test"]:
        os.makedirs(f"/opt/ml/processing/{name}", exist_ok=True)

    # 빌트인 XGBoost용: label을 맨 앞 열로, 헤더 없이 저장
    train = pd.concat([y_train, X_train], axis=1)
    val = pd.concat([y_val, X_val], axis=1)
    train.to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
    val.to_csv("/opt/ml/processing/validation/validation.csv", header=False, index=False)

    # 평가(배치 변환)용: 피처만(라벨 제외) / 정답 라벨은 따로 저장
    X_test.to_csv("/opt/ml/processing/test/test_x.csv", header=False, index=False)
    y_test.to_csv("/opt/ml/processing/test/test_y.csv", header=False, index=False)
    print("preprocessing done:", train.shape, val.shape, X_test.shape)


if __name__ == "__main__":
    main()'''),

    md("## 4. SageMaker Processing Job 실행\n\n`SKLearnProcessor` 는 scikit-learn이 설치된 관리형 컨테이너에서 위 스크립트를 실행합니다."),
    code2(
        r'''from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

processed = f"s3://{bucket}/{prefix}/processed"

# TODO: SKLearnProcessor를 생성하세요.
#  - framework_version="1.2-1", instance_type="ml.m5.xlarge", instance_count=1
#  - role=role, sagemaker_session=sess, base_job_name="student-preprocess"
#  - 참고: https://sagemaker.readthedocs.io/en/v2.219.0/frameworks/sklearn/sagemaker.sklearn.html
sklearn_processor = SKLearnProcessor(
    framework_version="____",
    role=role,
    instance_type="____",
    instance_count=1,
    base_job_name="student-preprocess",
    sagemaker_session=sess,
)

sklearn_processor.run(
    code="src/preprocessing.py",
    # TODO: 원본 데이터를 컨테이너의 /opt/ml/processing/input 으로 매핑하세요.
    inputs=[ProcessingInput(source=raw_s3, destination="____")],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"{processed}/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation", destination=f"{processed}/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"{processed}/test"),
    ],
)
print("done")''',
        r'''from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

processed = f"s3://{bucket}/{prefix}/processed"

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="student-preprocess",
    sagemaker_session=sess,
)

sklearn_processor.run(
    code="src/preprocessing.py",
    inputs=[ProcessingInput(source=raw_s3, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"{processed}/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation", destination=f"{processed}/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"{processed}/test"),
    ],
)
print("done")'''),

    md("## 5. 결과 경로 저장\n\n다음 노트북들이 사용할 S3 경로를 `%store` 로 저장합니다."),
    code(r'''train_s3 = f"s3://{bucket}/{prefix}/processed/train/train.csv"
validation_s3 = f"s3://{bucket}/{prefix}/processed/validation/validation.csv"
test_x_s3 = f"s3://{bucket}/{prefix}/processed/test/test_x.csv"
test_y_s3 = f"s3://{bucket}/{prefix}/processed/test/test_y.csv"

%store bucket prefix region train_s3 validation_s3 test_x_s3 test_y_s3
print("stored:")
for k in ["bucket", "prefix", "region", "train_s3", "validation_s3", "test_x_s3", "test_y_s3"]:
    print(" ", k, "=", eval(k))'''),

    md("✅ **완료!** 전처리된 데이터가 S3에 저장되었습니다. 다음은 `02_training.ipynb` — 빌트인 XGBoost로 모델을 학습합니다."),
])


# =============================================================================
# Notebook 02 - Training (built-in XGBoost)
# =============================================================================
add("02_training.ipynb", "training with built-in XGBoost", [
    md(r"""# 02. 모델 학습 — 빌트인 XGBoost

## 이 노트북에서 배우는 것
- SageMaker **빌트인 알고리즘 컨테이너**(XGBoost) 이미지를 가져오기
- `Estimator` 로 학습 작업(Training Job) 구성 — 하이퍼파라미터 지정
- `train` / `validation` 채널을 지정하고 `fit()` 실행
- 학습된 **모델 아티팩트(model.tar.gz)** 위치 확인

전처리 노트북(`01`)을 먼저 실행했어야 합니다."""),

    md("## 0. 환경 복원"),
    code(r'''import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
role = get_execution_role()
region = sess.boto_region_name

# 01 노트북에서 저장한 값 복원
%store -r bucket prefix train_s3 validation_s3 test_x_s3 test_y_s3
print("train:", train_s3)
print("valid:", validation_s3)'''),

    md("## 1. XGBoost 컨테이너 이미지 가져오기\n\nSageMaker는 알고리즘별 관리형 도커 이미지를 제공합니다. 리전에 맞는 이미지 URI를 조회합니다."),
    code2(
        r'''from sagemaker import image_uris

# TODO: XGBoost 빌트인 컨테이너 이미지를 조회하세요.
#  - framework 이름은 무엇일까요? version 은 "1.7-1" 을 사용합니다.
#  - 참고: https://sagemaker.readthedocs.io/en/v2.219.0/api/utility/image_uris.html
xgb_image_uri = image_uris.retrieve(framework="____", region=region, version="1.7-1")
print(xgb_image_uri)''',
        r'''from sagemaker import image_uris

xgb_image_uri = image_uris.retrieve(framework="xgboost", region=region, version="1.7-1")
print(xgb_image_uri)'''),

    md(r"""## 2. Estimator 구성

`Estimator` 는 "어떤 이미지를, 어떤 인스턴스에서, 어떤 하이퍼파라미터로 학습할지"를 정의합니다.

> 우리 문제는 3개 클래스 분류입니다. XGBoost에서 다중 클래스 확률을 출력하려면
> `objective="multi:softprob"` 과 `num_class` 를 지정해야 합니다."""),
    code2(
        r'''from sagemaker.estimator import Estimator

model_output = f"s3://{bucket}/{prefix}/models"

xgb = Estimator(
    image_uri=xgb_image_uri,
    role=role,
    instance_count=1,
    # TODO: 학습 인스턴스 타입을 지정하세요 (예: "ml.m5.xlarge").
    instance_type="____",
    output_path=model_output,
    sagemaker_session=sess,
    base_job_name="student-xgb",
)

# TODO: 다중 클래스(3개) 확률 출력을 위한 하이퍼파라미터를 채우세요.
#  - objective: "multi:softprob"
#  - num_class: 클래스 개수
#  - num_round: 부스팅 라운드 수 (예: 100)
#  - 참고: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
xgb.set_hyperparameters(
    objective="____",
    num_class=____,
    num_round=____,
    max_depth=5,
    eta=0.2,
    subsample=0.8,
    # TODO: 다중 클래스 분류에 적합한 평가 지표를 지정하세요.
    #  - 힌트: multi-class log loss
    eval_metric="____",
)''',
        r'''from sagemaker.estimator import Estimator

model_output = f"s3://{bucket}/{prefix}/models"

xgb = Estimator(
    image_uri=xgb_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=model_output,
    sagemaker_session=sess,
    base_job_name="student-xgb",
)

xgb.set_hyperparameters(
    objective="multi:softprob",
    num_class=3,
    num_round=100,
    max_depth=5,
    eta=0.2,
    subsample=0.8,
    eval_metric="mlogloss",
)'''),

    md("## 3. 학습 채널 지정 & 실행\n\n빌트인 XGBoost는 CSV 입력을 받습니다. `train`, `validation` 두 채널을 넘깁니다. (학습에 수 분 소요)"),
    code2(
        r'''from sagemaker.inputs import TrainingInput

# TODO: 입력 채널의 content_type 을 지정하세요 (CSV).
train_input = TrainingInput(train_s3, content_type="____")
val_input = TrainingInput(validation_s3, content_type="____")

# TODO: fit 에 넘기는 딕셔너리의 키가 곧 "채널 이름"입니다.
#       학습 채널 이름과 검증 채널 이름을 각각 지정하세요.
xgb.fit({"____": train_input, "____": val_input})''',
        r'''from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3, content_type="csv")
val_input = TrainingInput(validation_s3, content_type="csv")

xgb.fit({"train": train_input, "validation": val_input})'''),

    md("## 4. 학습 곡선 시각화\n\n`TrainingJobAnalytics` 는 CloudWatch Logs에서 매 라운드의 지표 값을 추출합니다. Debugger 없이도 학습/검증 오차의 변화를 확인할 수 있습니다."),
    code2(r'''import re
import boto3
import matplotlib.pyplot as plt
%matplotlib inline

# CloudWatch Logs에서 매 라운드의 지표를 추출합니다.
job_name = xgb.latest_training_job.name
logs_client = boto3.client("logs", region_name=region)

# TODO: SageMaker 학습 작업의 CloudWatch Logs 그룹 이름을 지정하세요.
#  - 힌트: "/aws/sagemaker/TrainingJobs"
log_group = "____"

# 로그 스트림 찾기
streams = logs_client.describe_log_streams(
    logGroupName=log_group,
    logStreamNamePrefix=job_name,
)["logStreams"]

train_loss, val_loss = [], []
pattern = re.compile(r"\[(\d+)\].*?train-mlogloss:([\d.]+).*?validation-mlogloss:([\d.]+)")

for stream in streams:
    token = None
    while True:
        kwargs = {"logGroupName": log_group, "logStreamName": stream["logStreamName"], "startFromHead": True}
        if token:
            kwargs["nextToken"] = token
        resp = logs_client.get_log_events(**kwargs)
        for event in resp["events"]:
            m = pattern.search(event["message"])
            if m:
                rnd = int(m.group(1))
                train_loss.append((rnd, float(m.group(2))))
                val_loss.append((rnd, float(m.group(3))))
        if resp["nextForwardToken"] == token:
            break
        token = resp["nextForwardToken"]

if not train_loss:
    print("WARNING: No per-round metrics found in logs yet. Wait a minute and re-run.")
else:
    train_loss.sort(); val_loss.sort()
    rounds_t, vals_t = zip(*train_loss)
    rounds_v, vals_v = zip(*val_loss)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds_t, vals_t, label="train mlogloss")
    ax.plot(rounds_v, vals_v, label="validation mlogloss")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("mlogloss")
    ax.set_title("Training Curve - train vs validation loss")
    ax.legend()
    plt.tight_layout()
    plt.show()''',
        r'''import re
import boto3
import matplotlib.pyplot as plt
%matplotlib inline

# CloudWatch Logs에서 매 라운드의 지표를 추출합니다.
job_name = xgb.latest_training_job.name
logs_client = boto3.client("logs", region_name=region)
log_group = "/aws/sagemaker/TrainingJobs"

# 로그 스트림 찾기
streams = logs_client.describe_log_streams(
    logGroupName=log_group,
    logStreamNamePrefix=job_name,
)["logStreams"]

train_loss, val_loss = [], []
pattern = re.compile(r"\[(\d+)\].*?train-mlogloss:([\d.]+).*?validation-mlogloss:([\d.]+)")

for stream in streams:
    token = None
    while True:
        kwargs = {"logGroupName": log_group, "logStreamName": stream["logStreamName"], "startFromHead": True}
        if token:
            kwargs["nextToken"] = token
        resp = logs_client.get_log_events(**kwargs)
        for event in resp["events"]:
            m = pattern.search(event["message"])
            if m:
                rnd = int(m.group(1))
                train_loss.append((rnd, float(m.group(2))))
                val_loss.append((rnd, float(m.group(3))))
        if resp["nextForwardToken"] == token:
            break
        token = resp["nextForwardToken"]

if not train_loss:
    print("WARNING: No per-round metrics found in logs yet. Wait a minute and re-run.")
else:
    train_loss.sort(); val_loss.sort()
    rounds_t, vals_t = zip(*train_loss)
    rounds_v, vals_v = zip(*val_loss)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds_t, vals_t, label="train mlogloss")
    ax.plot(rounds_v, vals_v, label="validation mlogloss")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("mlogloss")
    ax.set_title("Training Curve - train vs validation loss")
    ax.legend()
    plt.tight_layout()
    plt.show()'''),

    md("## 5. 학습 결과 저장\n\n학습 작업 이름과 모델 아티팩트(S3) 위치를 다음 노트북용으로 저장합니다."),
    code(r'''training_job_name = xgb.latest_training_job.name
model_data = xgb.model_data
print("training job :", training_job_name)
print("model.tar.gz :", model_data)

%store xgb_image_uri training_job_name model_data'''),

    md(r"""> 로그에서 `validation-mlogloss` 값이 라운드마다 감소하는 것을 확인해 보세요.
> 이 값이 뒤의 **하이퍼파라미터 튜닝(04)** 에서 최소화할 목표 지표가 됩니다.

✅ **완료!** 다음은 `03_evaluation.ipynb` — 테스트셋으로 모델 성능을 평가합니다."""),
])


# =============================================================================
# Notebook 03 - Evaluation (Batch Transform + metrics)
# =============================================================================
add("03_evaluation.ipynb", "evaluation via Batch Transform", [
    md(r"""# 03. 모델 평가 — Batch Transform

## 이 노트북에서 배우는 것
- 학습된 모델을 **Batch Transform**(대량 오프라인 추론)으로 테스트셋에 적용
- 다중 클래스 확률 출력(softprob)을 **예측 라벨**로 변환 (argmax)
- **혼동 행렬 / 클래스별 precision·recall·F1 / macro-F1** 로 성능 해석
- 왜 정확도(accuracy)만 보면 안 되는지 (클래스 불균형)

`02_training.ipynb` 를 먼저 실행했어야 합니다."""),

    md("## 0. 환경 복원\n\n학습 작업 이름으로 Estimator를 다시 붙여(attach) 학습된 모델을 재사용합니다."),
    code(r'''import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

sess = sagemaker.Session()
role = get_execution_role()
region = sess.boto_region_name

%store -r bucket prefix training_job_name test_x_s3 test_y_s3

# 이미 완료된 학습 작업에 다시 연결 -> 학습된 모델을 그대로 사용
xgb = Estimator.attach(training_job_name, sagemaker_session=sess)
print("attached:", training_job_name)'''),

    md("## 1. Batch Transform 실행\n\n`Transformer` 는 엔드포인트를 상시 띄우지 않고, 대량의 입력을 한 번에 채점한 뒤 리소스를 정리합니다. 테스트 **피처(라벨 없음)** 파일을 입력으로 줍니다."),
    code2(
        r'''batch_output = f"s3://{bucket}/{prefix}/batch-eval"

transformer = xgb.transformer(
    instance_count=1,
    # TODO: 배치 변환 인스턴스 타입을 지정하세요 (예: "ml.m5.xlarge").
    instance_type="____",
    output_path=batch_output,
    # TODO: 각 샘플의 예측 결과를 한 줄씩 조합하려면 어떻게 지정할까요?
    assemble_with="____",
    accept="text/csv",
)

# TODO: 입력은 CSV이고 한 줄이 한 샘플입니다. content_type 과 split_type 을 지정하세요.
#  - content_type="text/csv", split_type="Line"
#  - 참고: https://sagemaker.readthedocs.io/en/v2.219.0/api/inference/transformer.html
transformer.transform(test_x_s3, content_type="____", split_type="____")
transformer.wait()
print("batch output:", transformer.output_path)''',
        r'''batch_output = f"s3://{bucket}/{prefix}/batch-eval"

transformer = xgb.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=batch_output,
    assemble_with="Line",
    accept="text/csv",
)

transformer.transform(test_x_s3, content_type="text/csv", split_type="Line")
transformer.wait()
print("batch output:", transformer.output_path)'''),

    md("## 2. 예측 결과 내려받기 & 라벨 변환\n\nBatch Transform 출력은 입력 파일명에 `.out` 이 붙습니다. 각 줄은 3개 클래스 확률입니다."),
    code2(
        r'''import json
import numpy as np
import pandas as pd
from sagemaker.s3 import S3Downloader

S3Downloader.download(transformer.output_path, "batch_out")
S3Downloader.download(test_y_s3, "batch_out")

proba = np.array([json.loads(line) for line in open("batch_out/test_x.csv.out")])
y_true = pd.read_csv("batch_out/test_y.csv", header=None).values.ravel()

# TODO: 각 행에서 확률이 가장 큰 클래스의 인덱스를 예측 라벨로 만드세요.
#  - 힌트: numpy 의 argmax 를 axis=1 로 사용
y_pred = ____

print("samples:", len(y_true), "| pred shape:", proba.shape)
pd.DataFrame(proba, columns=["p_Dropout", "p_Enrolled", "p_Graduate"]).head()''',
        r'''import json
import numpy as np
import pandas as pd
from sagemaker.s3 import S3Downloader

S3Downloader.download(transformer.output_path, "batch_out")
S3Downloader.download(test_y_s3, "batch_out")

proba = np.array([json.loads(line) for line in open("batch_out/test_x.csv.out")])
y_true = pd.read_csv("batch_out/test_y.csv", header=None).values.ravel()

y_pred = np.argmax(proba, axis=1)

print("samples:", len(y_true), "| pred shape:", proba.shape)
pd.DataFrame(proba, columns=["p_Dropout", "p_Enrolled", "p_Graduate"]).head()'''),

    md("## 3. 성능 지표\n\n혼동 행렬과 클래스별 리포트를 확인합니다."),
    code2(
        r'''from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

labels = ["Dropout", "Enrolled", "Graduate"]

print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(confusion_matrix(y_true, y_pred), index=labels, columns=labels))
print()
print(classification_report(y_true, y_pred, target_names=labels, digits=3))

acc = accuracy_score(y_true, y_pred)
# TODO: 클래스 불균형에서는 macro 평균 F1이 더 공정한 지표입니다. average 인자를 지정하세요.
macro_f1 = f1_score(y_true, y_pred, average="____")
print(f"accuracy : {acc:.3f}")
print(f"macro F1 : {macro_f1:.3f}")''',
        r'''from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

labels = ["Dropout", "Enrolled", "Graduate"]

print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(confusion_matrix(y_true, y_pred), index=labels, columns=labels))
print()
print(classification_report(y_true, y_pred, target_names=labels, digits=3))

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"accuracy : {acc:.3f}")
print(f"macro F1 : {macro_f1:.3f}")'''),

    md(r"""### 해석 포인트
- `Enrolled`(소수 클래스)의 **recall**이 다른 클래스보다 낮게 나오는지 확인하세요. 전체 정확도가
  높아도 소수 클래스를 잘 못 맞히는 경우가 흔합니다.
- 그래서 **macro-F1**(클래스별 F1의 단순 평균)을 함께 봅니다. 다음 노트북(`04`)에서
  하이퍼파라미터 튜닝으로 이 지표를 끌어올려 봅니다.

✅ **완료!** 다음은 `04_tuning.ipynb`."""),
])


# =============================================================================
# Notebook 04 - Hyperparameter tuning (Hyperparameter Optimization)
# =============================================================================
add("04_tuning.ipynb", "hyperparameter tuning (Hyperparameter Optimization)", [
    md(r"""# 04. 하이퍼파라미터 튜닝 (Automatic Model Tuning)

## 이 노트북에서 배우는 것
- **탐색 범위(search space)** 정의 — 정수/연속 파라미터
- `HyperparameterTuner` 로 **베이지안 최적화** 기반 자동 튜닝 구성
- 최적화 목표 지표(`validation:mlogloss`) 와 방향(최소화) 지정
- 튜닝 결과 분석 및 **최적 모델** 선택

`01`, `02` 노트북을 먼저 실행했어야 합니다."""),

    md("## 0. 환경 복원 & 튜닝 대상 Estimator 구성"),
    code(r'''import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

sess = sagemaker.Session()
role = get_execution_role()
region = sess.boto_region_name

%store -r bucket prefix train_s3 validation_s3 xgb_image_uri

xgb = Estimator(
    image_uri=xgb_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/{prefix}/models-hpo",
    sagemaker_session=sess,
    base_job_name="student-xgb-hpo",
)
# 튜닝하지 않는(고정) 하이퍼파라미터
xgb.set_hyperparameters(objective="multi:softprob", num_class=3, num_round=100, eval_metric="mlogloss")
print("estimator ready")'''),

    md(r"""## 1. 탐색 범위 정의

각 하이퍼파라미터를 하나의 값이 아니라 **범위**로 줍니다. 튜너가 이 공간을 탐색합니다.
- `IntegerParameter(min, max)` — 정수형
- `ContinuousParameter(min, max)` — 실수형"""),
    code2(
        r'''from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

# TODO: 아래 범위를 완성하세요. (권장 예시는 힌트를 참고)
#  - max_depth: 정수 3 ~ 8
#  - eta(학습률): 연속 0.05 ~ 0.4
#  - 참고: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
hyperparameter_ranges = {
    "max_depth": IntegerParameter(____, ____),
    "eta": ContinuousParameter(____, ____),
    "subsample": ContinuousParameter(0.6, 1.0),
    "min_child_weight": IntegerParameter(1, 6),
}
hyperparameter_ranges''',
        r'''from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

hyperparameter_ranges = {
    "max_depth": IntegerParameter(3, 8),
    "eta": ContinuousParameter(0.05, 0.4),
    "subsample": ContinuousParameter(0.6, 1.0),
    "min_child_weight": IntegerParameter(1, 6),
}
hyperparameter_ranges'''),

    md(r"""## 2. Tuner 구성

`validation:mlogloss` 는 **작을수록 좋은** 지표입니다. 목표 방향을 올바르게 지정하세요.
(빌트인 XGBoost는 지표 정의가 내장되어 있어 `metric_definitions` 를 따로 주지 않아도 됩니다.)"""),
    code2(
        r'''# TODO: Hyperparameter Optimization가 최적화할 목표 지표를 지정하세요.
#  - 힌트: 학습 시 eval_metric 으로 사용한 검증 지표와 동일합니다. (형식: "채널:지표명")
objective_metric_name = "____"

tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name=objective_metric_name,
    # TODO: mlogloss 는 작을수록 좋습니다. 목표 방향을 지정하세요 ("Minimize" / "Maximize").
    objective_type="____",
    hyperparameter_ranges=hyperparameter_ranges,
    # TODO: 총 학습 작업 수를 지정하세요 (워크샵에서는 8 정도 권장).
    max_jobs=____,
    max_parallel_jobs=2,
    base_tuning_job_name="student-hpo",
)
print("tuner ready")''',
        r'''objective_metric_name = "validation:mlogloss"

tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name=objective_metric_name,
    objective_type="Minimize",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=8,
    max_parallel_jobs=2,
    base_tuning_job_name="student-hpo",
)
print("tuner ready")'''),

    md("## 3. 튜닝 실행\n\n여러 학습 작업이 순차/병렬로 실행됩니다 (10~20분 소요될 수 있습니다)."),
    code2(
        r'''from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3, content_type="csv")
val_input = TrainingInput(validation_s3, content_type="csv")

# TODO: 학습 때와 동일하게 train/validation 채널을 넘겨 튜닝을 시작하세요.
tuner.fit({"____": train_input, "____": val_input})
tuner.wait()
print("tuning done")''',
        r'''from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3, content_type="csv")
val_input = TrainingInput(validation_s3, content_type="csv")

tuner.fit({"train": train_input, "validation": val_input})
tuner.wait()
print("tuning done")'''),

    md("## 4. 결과 분석 & 최적 모델 저장"),
    code(r'''import pandas as pd

df = tuner.analytics().dataframe()
cols = [c for c in ["max_depth", "eta", "subsample", "min_child_weight",
                     "FinalObjectiveValue", "TrainingJobName"] if c in df.columns]
display(df[cols].sort_values("FinalObjectiveValue").head(10))

best_training_job = tuner.best_training_job()
best_estimator = tuner.best_estimator()
best_model_data = best_estimator.model_data
print("best training job :", best_training_job)
print("best model.tar.gz :", best_model_data)

%store best_training_job best_model_data'''),

    md("## 5. 최적 모델 학습 곡선\n\n최적 학습 작업의 CloudWatch Logs에서 매 라운드별 train/validation loss를 시각화합니다."),
    code(r'''import re
import boto3
import matplotlib.pyplot as plt
%matplotlib inline

logs_client = boto3.client("logs", region_name=region)
log_group = "/aws/sagemaker/TrainingJobs"

streams = logs_client.describe_log_streams(
    logGroupName=log_group,
    logStreamNamePrefix=best_training_job,
)["logStreams"]

train_loss, val_loss = [], []
pattern = re.compile(r"\[(\d+)\].*?train-mlogloss:([\d.]+).*?validation-mlogloss:([\d.]+)")

for stream in streams:
    token = None
    while True:
        kwargs = {"logGroupName": log_group, "logStreamName": stream["logStreamName"], "startFromHead": True}
        if token:
            kwargs["nextToken"] = token
        resp = logs_client.get_log_events(**kwargs)
        for event in resp["events"]:
            m = pattern.search(event["message"])
            if m:
                rnd = int(m.group(1))
                train_loss.append((rnd, float(m.group(2))))
                val_loss.append((rnd, float(m.group(3))))
        if resp["nextForwardToken"] == token:
            break
        token = resp["nextForwardToken"]

if not train_loss:
    print("WARNING: No per-round metrics found. Wait a minute and re-run.")
else:
    train_loss.sort(); val_loss.sort()
    rounds_t, vals_t = zip(*train_loss)
    rounds_v, vals_v = zip(*val_loss)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds_t, vals_t, label="train mlogloss")
    ax.plot(rounds_v, vals_v, label="validation mlogloss")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("mlogloss")
    ax.set_title(f"Best Model Training Curve ({best_training_job})")
    ax.legend()
    plt.tight_layout()
    plt.show()'''),

    md("✅ **완료!** 최적 모델을 골랐습니다. 다음은 `05_deployment.ipynb` — 실시간 엔드포인트로 배포합니다."),
])


# =============================================================================
# Notebook 05 - Deployment (real-time endpoint)
# =============================================================================
add("05_deployment.ipynb", "deploy to real-time endpoint", [
    md(r"""# 05. 실시간 엔드포인트 배포

## 이 노트북에서 배우는 것
- 최적 모델(튜닝 결과)로 **실시간 추론 엔드포인트** 생성
- 요청/응답 직렬화기(serializer/deserializer) 설정
- 배포된 엔드포인트로 **스모크 테스트**

> ⚠️ **비용 주의**: 실시간 엔드포인트는 삭제 전까지 **시간당 과금**됩니다.
> 워크샵이 끝나면 `08_cleanup.ipynb` 를 반드시 실행하세요.

`04_tuning.ipynb` 를 먼저 실행했어야 합니다."""),

    md("## 0. 환경 복원 & 모델 재구성\n\n튜닝에서 고른 최적 모델 아티팩트로 배포용 `Model` 객체를 만듭니다."),
    code(r'''import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model

sess = sagemaker.Session()
role = get_execution_role()

%store -r bucket prefix xgb_image_uri best_model_data

model = Model(
    image_uri=xgb_image_uri,
    model_data=best_model_data,
    role=role,
    sagemaker_session=sess,
)
print("model artifact:", best_model_data)'''),

    md("## 1. 엔드포인트 배포\n\n`model.deploy()` 는 내부적으로 **Model 등록 → EndpointConfig 생성 → Endpoint 생성**을 한 번에 처리합니다.\n배포에는 수 분이 걸립니다. CSV로 요청/응답을 주고받도록 직렬화기를 지정합니다."),
    code2(
        r'''from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
import time, boto3

# TODO: 엔드포인트 이름을 지정하세요 (예: "student-success-endpoint").
#  - 이 이름은 이후 노트북(06, 07, 08)에서도 사용됩니다.
endpoint_name = "____"

model.deploy(
    # TODO: 엔드포인트에 배치할 초기 인스턴스 수를 지정하세요.
    initial_instance_count=____,
    # TODO: 실시간 추론용 인스턴스 타입을 지정하세요 (예: "ml.m5.large").
    instance_type="____",
    endpoint_name=endpoint_name,
    wait=False,
)

sm_client = boto3.client("sagemaker")
print("deploying...", end="")
while True:
    status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
    if status == "InService":
        print(" deployed!")
        break
    elif status == "Failed":
        print(" FAILED!")
        raise RuntimeError("Endpoint deployment failed")
    print(".", end="", flush=True)
    time.sleep(30)

# wait=False 시 predictor가 None이므로 직접 생성합니다.
from sagemaker.predictor import Predictor
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sess,
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer(),
)''',
        r'''from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
import time, boto3

endpoint_name = "student-success-endpoint"

model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    wait=False,
)

sm_client = boto3.client("sagemaker")
print("deploying...", end="")
while True:
    status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
    if status == "InService":
        print(" deployed!")
        break
    elif status == "Failed":
        print(" FAILED!")
        raise RuntimeError("Endpoint deployment failed")
    print(".", end="", flush=True)
    time.sleep(30)

# wait=False 시 predictor가 None이므로 직접 생성합니다.
from sagemaker.predictor import Predictor
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sess,
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer(),
)'''),

    md(r"""## 2. 스모크 테스트

테스트셋에서 한 샘플을 꺼내 엔드포인트에 보내 봅니다.

> **스모크 테스트 vs A/B 테스트**
> - **스모크 테스트**: 배포 직후 "엔드포인트가 정상 응답하는지" 를 소수의 샘플로 빠르게 확인하는 것입니다.
>   기능적 정상 동작(응답 형식, 에러 없음, 합리적인 값)을 검증하는 **sanity check** 입니다.
> - **A/B 테스트**: 실제 트래픽의 일부를 새 모델로 보내고, 기존 모델과 **비즈니스 지표(전환율, 정확도 등)를 통계적으로 비교**하는 것입니다.
>   수일~수주간 실 사용자 트래픽으로 진행합니다.
>
> 즉, 스모크 테스트는 "작동하는가?"를, A/B 테스트는 "더 나은가?"를 답합니다."""),
    code2(
        r'''import numpy as np
import pandas as pd
from sagemaker.s3 import S3Downloader

%store -r test_x_s3
S3Downloader.download(test_x_s3, "smoke")
X = pd.read_csv("smoke/test_x.csv", header=None)
sample = X.iloc[0].tolist()

# CSVSerializer가 리스트를 CSV 한 줄로 변환해 전송합니다.
resp = predictor.predict(sample)          # -> [['0.1', '0.2', '0.7']]
proba = [float(p) for p in resp[0]]
classes = ["Dropout", "Enrolled", "Graduate"]

# TODO: 확률이 가장 큰 클래스를 예측 라벨로 고르세요. (힌트: int(np.argmax(proba)))
pred_idx = ____
print("probabilities:", dict(zip(classes, [round(p, 3) for p in proba])))
print("prediction   :", classes[pred_idx])''',
        r'''import numpy as np
import pandas as pd
from sagemaker.s3 import S3Downloader

%store -r test_x_s3
S3Downloader.download(test_x_s3, "smoke")
X = pd.read_csv("smoke/test_x.csv", header=None)
sample = X.iloc[0].tolist()

# CSVSerializer가 리스트를 CSV 한 줄로 변환해 전송합니다.
resp = predictor.predict(sample)          # -> [['0.1', '0.2', '0.7']]
proba = [float(p) for p in resp[0]]
classes = ["Dropout", "Enrolled", "Graduate"]

pred_idx = int(np.argmax(proba))
print("probabilities:", dict(zip(classes, [round(p, 3) for p in proba])))
print("prediction   :", classes[pred_idx])'''),

    md("## 3. 엔드포인트 이름 저장"),
    code(r'''%store endpoint_name
print("stored endpoint_name =", endpoint_name)'''),

    md(r"""> 🔒 **보안 메모**: SageMaker 엔드포인트는 공개(public) URL이 아니라 **AWS IAM/SigV4 서명**으로만
> 호출할 수 있는 프라이빗 엔드포인트입니다. 뒤의 웹앱(`07`)도 IAM 자격증명으로 호출합니다.

✅ **완료!** 엔드포인트가 살아 있습니다. 다음은 `06_inference.ipynb` — 실시간 추론."""),
])


# =============================================================================
# Notebook 06 - Inference (real-time + boto3) + cleanup  [open-ended]
# =============================================================================
add("06_inference.ipynb", "inference (open-ended)", [
    md(r"""# 06. 추론

> 이 노트북은 **개방형**입니다. 앞에서 익힌 패턴을 응용해 직접 코드를 작성해 보세요.
> 힌트와 문서 링크만 제공됩니다.

## 이 노트북에서 배우는 것
- 배포된 엔드포인트를 **SageMaker SDK Predictor** 로 호출하는 헬퍼 함수 작성
- **boto3 `sagemaker-runtime`** 로 직접 호출 (웹앱이 사용하는 저수준 방식)

`05_deployment.ipynb` 를 먼저 실행해 엔드포인트가 떠 있어야 합니다."""),

    md("## 0. 환경 복원 & Predictor 재생성"),
    code(r'''import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker.s3 import S3Downloader
import pandas as pd

sess = sagemaker.Session()
%store -r endpoint_name region test_x_s3

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sess,
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer(),
)

S3Downloader.download(test_x_s3, "infer")
X = pd.read_csv("infer/test_x.csv", header=None)
sample = X.iloc[0].tolist()
print("endpoint:", endpoint_name)
print("sample features:", sample[:6], "...")'''),

    md(r"""## 1. 실시간 추론 헬퍼 함수 (직접 작성)

아래 `predict_student(features)` 를 완성하세요. 입력은 피처 리스트, 출력은 `(예측_라벨, 확률_딕셔너리)` 입니다.

**단계**
1. `predictor.predict(features)` 로 응답을 받습니다. 응답은 `[['p0','p1','p2']]` 형태입니다.
2. `resp[0]` 을 `float` 리스트로 변환합니다.
3. `np.argmax` 로 가장 큰 확률의 인덱스를 구해 `classes[idx]` 를 라벨로 만듭니다.

참고: https://sagemaker.readthedocs.io/en/v2.219.0/api/inference/predictors.html"""),
    code2(
        r'''import numpy as np
classes = ["Dropout", "Enrolled", "Graduate"]

def predict_student(features):
    # TODO: 위 3단계를 구현하세요.
    ____

# 검증
label, proba = predict_student(sample)
print("prediction:", label)
print("proba     :", proba)''',
        r'''import numpy as np
classes = ["Dropout", "Enrolled", "Graduate"]

def predict_student(features):
    resp = predictor.predict(features)
    proba = [float(p) for p in resp[0]]
    idx = int(np.argmax(proba))
    return classes[idx], dict(zip(classes, [round(p, 4) for p in proba]))

# 검증
label, proba = predict_student(sample)
print("prediction:", label)
print("proba     :", proba)'''),

    md(r"""## 2. boto3 로 직접 호출 (저수준 방식)

SDK 없이 `sagemaker-runtime` 클라이언트로 호출하는 방법입니다. **웹 애플리케이션(07)** 이 이 방식을 씁니다.
`ContentType` 과 `Body`(CSV 문자열)를 채워 완성하세요.

참고: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html"""),
    code2(
        r'''import boto3

runtime = boto3.client("sagemaker-runtime", region_name=region)
csv_row = ",".join(str(x) for x in sample)

# TODO: invoke_endpoint 호출을 완성하세요.
resp = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="____",
    Body=____,
)
print(resp["Body"].read().decode())''',
        r'''import boto3

runtime = boto3.client("sagemaker-runtime", region_name=region)
csv_row = ",".join(str(x) for x in sample)

resp = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=csv_row,
)
print(resp["Body"].read().decode())'''),

    md(r"""✅ **핵심 워크플로우 완주!** 전처리 → 학습 → 평가 → 튜닝 → 배포 → 추론까지 마쳤습니다.

### 다음 단계
- `07_web_app.ipynb` — (선택) 이 엔드포인트를 호출하는 **웹 애플리케이션**을 만들어 로컬 개발 서버로 띄워 봅니다.
- `08_cleanup.ipynb` — 모든 리소스를 정리합니다. **워크샵이 끝나면 반드시 실행하세요.**"""),
])



# =============================================================================
# Notebook 07 (optional) - Gradio web app (runs in notebook cell)
# =============================================================================

_GRADIO_CODE_SOLUTION = r'''import json
import boto3
import numpy as np
import gradio as gr

%store -r endpoint_name region

CLASSES = ["Dropout", "Enrolled", "Graduate"]
runtime = boto3.client("sagemaker-runtime", region_name=region)

FEATURES = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
    "Nacionality", "Mother\'s qualification", "Father\'s qualification", "Mother\'s occupation",
    "Father\'s occupation", "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment",
    "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]

DEFAULTS = {
    "Marital status": 1, "Application mode": 17, "Application order": 1, "Course": 9238,
    "Daytime/evening attendance": 1, "Previous qualification": 1, "Previous qualification (grade)": 133.1,
    "Nacionality": 1, "Mother\'s qualification": 19, "Father\'s qualification": 19,
    "Mother\'s occupation": 5, "Father\'s occupation": 7, "Admission grade": 126.1,
    "Displaced": 1, "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
    "Gender": 0, "Scholarship holder": 0, "Age at enrollment": 20, "International": 0,
    "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8, "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 12.286, "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 8, "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 12.2, "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 11.1, "Inflation rate": 1.4, "GDP": 0.32,
}


def call_endpoint(feature_row):
    csv = ",".join(str(x) for x in feature_row)
    resp = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="text/csv", Body=csv)
    text = resp["Body"].read().decode().strip()
    proba = [float(p) for p in text.split(",")]
    return proba


def predict(age, admission, prev_grade, gender, s1_approved, s1_grade, s2_approved, s2_grade,
            tuition, scholarship, debtor):
    overrides = {
        "Age at enrollment": age, "Admission grade": admission,
        "Previous qualification (grade)": prev_grade, "Gender": gender,
        "Curricular units 1st sem (approved)": s1_approved,
        "Curricular units 1st sem (grade)": s1_grade,
        "Curricular units 2nd sem (approved)": s2_approved,
        "Curricular units 2nd sem (grade)": s2_grade,
        "Tuition fees up to date": int(tuition), "Scholarship holder": int(scholarship),
        "Debtor": int(debtor),
    }
    values = dict(DEFAULTS)
    values.update(overrides)
    feature_row = [values[f] for f in FEATURES]
    proba = call_endpoint(feature_row)
    pred = CLASSES[int(np.argmax(proba))]
    result = {CLASSES[i]: round(proba[i], 4) for i in range(3)}
    return pred, result


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(17, 60, value=20, step=1, label="Age at enrollment"),
        gr.Slider(0, 200, value=126.1, step=0.1, label="Admission grade"),
        gr.Slider(0, 200, value=133.1, step=0.1, label="Previous qualification grade"),
        gr.Radio(["Female (0)", "Male (1)"], value="Female (0)", label="Gender", type="index"),
        gr.Slider(0, 26, value=5, step=1, label="1st sem approved units"),
        gr.Slider(0, 20, value=12.3, step=0.1, label="1st sem grade"),
        gr.Slider(0, 26, value=5, step=1, label="2nd sem approved units"),
        gr.Slider(0, 20, value=12.2, step=0.1, label="2nd sem grade"),
        gr.Checkbox(value=True, label="Tuition fees up to date"),
        gr.Checkbox(value=False, label="Scholarship holder"),
        gr.Checkbox(value=False, label="Debtor"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Class probabilities"),
    ],
    title="Student Success Prediction",
    description=f"SageMaker Endpoint: {endpoint_name} | Region: {region}",
)
demo.launch(share=True)'''

_GRADIO_CODE_BLANK = r'''import json
import boto3
import numpy as np
import gradio as gr

%store -r endpoint_name region

CLASSES = ["Dropout", "Enrolled", "Graduate"]
runtime = boto3.client("sagemaker-runtime", region_name=region)

FEATURES = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
    "Nacionality", "Mother\'s qualification", "Father\'s qualification", "Mother\'s occupation",
    "Father\'s occupation", "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment",
    "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]

DEFAULTS = {
    "Marital status": 1, "Application mode": 17, "Application order": 1, "Course": 9238,
    "Daytime/evening attendance": 1, "Previous qualification": 1, "Previous qualification (grade)": 133.1,
    "Nacionality": 1, "Mother\'s qualification": 19, "Father\'s qualification": 19,
    "Mother\'s occupation": 5, "Father\'s occupation": 7, "Admission grade": 126.1,
    "Displaced": 1, "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
    "Gender": 0, "Scholarship holder": 0, "Age at enrollment": 20, "International": 0,
    "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8, "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 12.286, "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 8, "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 12.2, "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 11.1, "Inflation rate": 1.4, "GDP": 0.32,
}


def call_endpoint(feature_row):
    # TODO: 06 노트북의 boto3 호출 방식을 응용해 엔드포인트를 호출하세요.
    # feature_row(리스트)를 CSV 문자열로 변환 -> invoke_endpoint -> 응답 파싱 -> 확률 리스트 반환
    raise NotImplementedError("call_endpoint 를 구현하세요")


def predict(age, admission, prev_grade, gender, s1_approved, s1_grade, s2_approved, s2_grade,
            tuition, scholarship, debtor):
    overrides = {
        "Age at enrollment": age, "Admission grade": admission,
        "Previous qualification (grade)": prev_grade, "Gender": gender,
        "Curricular units 1st sem (approved)": s1_approved,
        "Curricular units 1st sem (grade)": s1_grade,
        "Curricular units 2nd sem (approved)": s2_approved,
        "Curricular units 2nd sem (grade)": s2_grade,
        "Tuition fees up to date": int(tuition), "Scholarship holder": int(scholarship),
        "Debtor": int(debtor),
    }
    values = dict(DEFAULTS)
    values.update(overrides)
    feature_row = [values[f] for f in FEATURES]
    proba = call_endpoint(feature_row)
    pred = CLASSES[int(np.argmax(proba))]
    result = {CLASSES[i]: round(proba[i], 4) for i in range(3)}
    return pred, result


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(17, 60, value=20, step=1, label="Age at enrollment"),
        gr.Slider(0, 200, value=126.1, step=0.1, label="Admission grade"),
        gr.Slider(0, 200, value=133.1, step=0.1, label="Previous qualification grade"),
        gr.Radio(["Female (0)", "Male (1)"], value="Female (0)", label="Gender", type="index"),
        gr.Slider(0, 26, value=5, step=1, label="1st sem approved units"),
        gr.Slider(0, 20, value=12.3, step=0.1, label="1st sem grade"),
        gr.Slider(0, 26, value=5, step=1, label="2nd sem approved units"),
        gr.Slider(0, 20, value=12.2, step=0.1, label="2nd sem grade"),
        gr.Checkbox(value=True, label="Tuition fees up to date"),
        gr.Checkbox(value=False, label="Scholarship holder"),
        gr.Checkbox(value=False, label="Debtor"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Class probabilities"),
    ],
    title="Student Success Prediction",
    description=f"SageMaker Endpoint: {endpoint_name} | Region: {region}",
)
demo.launch(share=True)'''

add("07_web_app.ipynb", "optional: Gradio web app", [
    md(r"""# 07. (선택 과제) 실시간 추론 웹 애플리케이션

> **선택/심화 과제입니다.** 배포한 SageMaker 엔드포인트를 호출하는 **웹앱**을 만들어 봅니다.
> **Gradio**를 사용하며 `share=True` 옵션으로 공개 URL이 자동 생성되어 별도 포트/프록시 설정 없이 바로 접속할 수 있습니다.

## 아키텍처
```
브라우저 ──HTTPS──> Gradio 공개 URL ──> 노트북 내 Gradio 서버 ──boto3/SigV4──> SageMaker 엔드포인트
```
- 웹앱은 폼 입력을 36개 피처 벡터로 조립해 엔드포인트에 보내고, 3개 클래스 확률을 받아 표시합니다.
- 엔드포인트는 공개되지 않으며 **AWS 자격증명(IAM)** 으로만 호출됩니다.

## 전제
`05_deployment.ipynb` 로 엔드포인트가 떠 있어야 합니다."""),

    md("## 0. Gradio 설치 & 엔드포인트 상태 확인"),
    code(r'''import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "-qU"],
                      stdout=subprocess.DEVNULL)
print("gradio installed")

import boto3, sagemaker
sess = sagemaker.Session()
%store -r endpoint_name region
sm = boto3.client("sagemaker", region_name=region)
status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
print(f"{endpoint_name} -> {status}")
assert status == "InService", "엔드포인트가 InService 상태가 아닙니다."'''),

    md(r"""## 1. 웹앱 실행

아래 셀에서 `call_endpoint` 함수를 완성하세요. 나머지 UI 코드는 제공됩니다.
셀을 실행하면 Gradio가 공개 URL을 출력합니다. 해당 URL을 브라우저에서 열면 됩니다.

> `share=True`로 생성된 URL은 **72시간** 동안 유효하며 노트북 커널이 살아 있는 동안만 작동합니다."""),
    code2(_GRADIO_CODE_BLANK, _GRADIO_CODE_SOLUTION),

    md("## 2. 종료\n\n테스트가 끝나면 아래 셀을 실행해 Gradio 서버를 종료합니다."),
    code(r'''demo.close()
print("Gradio server stopped")'''),

    md(r"""🎉 **수고하셨습니다!** 데이터 전처리부터 학습·평가·튜닝·배포·추론, 그리고 웹앱까지 완성했습니다.

> 워크샵이 끝나면 `08_cleanup.ipynb` 를 실행해 모든 리소스를 정리하세요."""),
])

# Notebook 08 - Cleanup (delete all resources)
# =============================================================================
add("08_cleanup.ipynb", "resource cleanup", [
    md(r"""# 08. 🧹 리소스 정리

> **워크샵이 끝나면 반드시 이 노트북을 실행하세요.** 실시간 엔드포인트는 삭제 전까지 시간당 과금됩니다.

이 노트북은 워크샵에서 생성한 모든 SageMaker 리소스와 S3 데이터를 삭제합니다."""),

    md("## 0. 환경 복원"),
    code(r'''import boto3
import sagemaker

sess = sagemaker.Session()
%store -r bucket prefix endpoint_name region

sm = boto3.client("sagemaker", region_name=region)
s3 = boto3.resource("s3", region_name=region)
print("region  :", region)
print("bucket  :", bucket)
print("prefix  :", prefix)
print("endpoint:", endpoint_name)'''),

    md("## 1. 엔드포인트 삭제\n\nEndpoint → Endpoint Configuration → Model 순으로 삭제합니다."),
    code(r'''# 엔드포인트 삭제
try:
    sm.delete_endpoint(EndpointName=endpoint_name)
    print("deleted endpoint:", endpoint_name)
except Exception as e:
    print("endpoint:", e)

# 엔드포인트 구성 삭제
try:
    sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("deleted endpoint config:", endpoint_name)
except Exception as e:
    print("endpoint config:", e)'''),

    md("## 2. 모델 삭제\n\n배포용 모델뿐 아니라, Hyperparameter Optimization 튜닝에서 생성된 모든 모델을 삭제합니다."),
    code(r'''# 워크샵에서 생성한 모든 모델 삭제 (student-xgb, student-xgb-hpo 등)
search_prefixes = ["student-xgb", "student-success"]
deleted_models = 0
for name_prefix in search_prefixes:
    paginator = sm.get_paginator("list_models")
    for page in paginator.paginate(NameContains=name_prefix):
        for m in page["Models"]:
            sm.delete_model(ModelName=m["ModelName"])
            deleted_models += 1
            print("  deleted model:", m["ModelName"])
print(f"total models deleted: {deleted_models}")'''),

    md("## 3. 학습 작업 / 튜닝 작업 정리\n\n학습·튜닝 작업 자체는 삭제 API가 없지만(기록은 자동 보관), **남은 Endpoint Config**가 있으면 삭제합니다."),
    code(r'''# 남은 엔드포인트 구성 삭제
paginator = sm.get_paginator("list_endpoint_configs")
deleted_configs = 0
for page in paginator.paginate(NameContains="student"):
    for cfg in page["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        deleted_configs += 1
        print("  deleted config:", cfg["EndpointConfigName"])
print(f"total endpoint configs deleted: {deleted_configs}")'''),

    md("## 4. S3 워크샵 데이터 삭제\n\n워크샵에서 사용한 S3 prefix 하위의 모든 객체를 삭제합니다."),
    code(r'''bucket_obj = s3.Bucket(bucket)
objects = list(bucket_obj.objects.filter(Prefix=f"{prefix}/"))
if objects:
    bucket_obj.delete_objects(Delete={"Objects": [{"Key": o.key} for o in objects]})
    print(f"deleted {len(objects)} objects from s3://{bucket}/{prefix}/")
else:
    print("no objects to delete")'''),

    md(r"""✅ **정리 완료!**

남은 리소스가 없는지 SageMaker 콘솔에서 확인하세요:
- **Inference > Endpoints** — 활성 엔드포인트 없는지
- **Inference > Models** — 등록된 모델 없는지
- **S3 콘솔** — 워크샵 버킷/prefix 삭제 확인

수고하셨습니다! 🎉"""),
])


# =============================================================================
if __name__ == "__main__":
    build()
