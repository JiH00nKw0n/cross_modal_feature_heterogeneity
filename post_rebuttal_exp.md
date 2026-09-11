# 후속 측정 실행 안내 (post_rebuttal)

이 문서 하나만 보고 처음부터 끝까지 실행할 수 있게 썼음. 코드를 읽지 않아도 되고,
다른 파일을 열어보지 않아도 됨. 명령은 모두 그대로 복사해서 쓰면 됨.

---

## 1. 이 실험이 무엇이고 무엇이 나오는가

한 번 실행하면 논문에 들어갈 표 하나, 그림 하나, 그리고 후속 측정 11가지 보고서가
모두 나옴.

먼저 용어를 정리함.

- **희소 오토인코더(sparse autoencoder)** 는 벡터 하나를 받아서 아주 적은 수의
  좌표만 켜지는 형태로 다시 표현하도록 학습한 신경망임. 이 실험에서는 좌표를 총
  8192개 두고, 입력 하나당 켜지는 좌표를 8개 또는 32개로 제한함.
- **잠재 좌표(latent)** 는 그 8192개 좌표축 하나하나를 말함. 좌표 하나가 하나의
  개념에 대응한다고 보는 것이 이 연구의 전제임.
- **벡터 표현(embedding)** 은 사진 한 장이나 문장 하나를 CLIP 같은 시각 언어 모델에
  넣어서 얻은 숫자 벡터임. 이 실험은 사진과 문장을 각각 벡터로 바꾼 다음, 그 벡터
  위에서만 학습함. 원본 이미지 파일은 한 번 읽고 나면 다시 쓰지 않음.
- **공동 활성 행렬(panel.npz)** 은 이미지 쪽 좌표와 문장 쪽 좌표가 같은 입력 쌍에서
  함께 켜지는 정도를 모든 좌표 조합에 대해 계산해 저장한 파일임. 어느 이미지 좌표가
  어느 문장 좌표와 짝인지도 이 파일 안에 함께 들어 있음.

결과물은 세 가지임.

- **표 1**은 다섯 가지 방법의 복원 오차, 이미지와 문장 사이 검색 정확도,
  ImageNet 무학습 분류 정확도를 한 표에 모은 것임. 파일은
  `outputs/post_rebuttal/cc3m_clip_b32/table1.md` 와 같은 내용의 LaTeX 판
  `outputs/post_rebuttal/cc3m_clip_b32/table1.tex` 임.
- **그림 2**는 두 좌표가 함께 켜지는 정도별로 두 좌표의 방향이 얼마나 벌어져 있는지
  분포를 그린 것임. 파일은 `outputs/post_rebuttal/coco_clip_b32/multi_density.pdf`
  이고, 같은 수치를 표로 적은 `outputs/post_rebuttal/coco_clip_b32/figure2_bin_stats.md`
  가 옆에 함께 생김.
- **후속 측정 11가지**는 각각 보고서 한 개와 수치 파일 한 개로 나옴. 위치는
  `outputs/post_rebuttal/rebuttal/coco_k8/` 와 `outputs/post_rebuttal/rebuttal/cc3m_k32/`
  임. 11가지 중 8가지는 두 조건 모두에서 돌고, 3가지는 CC3M 조건에서만 돎. 상관 구간별
  거리 측정은 COCO 조건에서는 그림 2가 같은 수치를 이미 쓰기 때문이고, 검색 정확도를
  쓰는 두 측정은 COCO로 학습한 모델에게 COCO 검색은 처음 보는 자료가 아니기 때문임.
  이 모든 것을 한 문서로 묶은 것이 `outputs/post_rebuttal/post_rebuttal_results.md`
  이고, 이 파일 하나만 열어도 모든 수치를 읽을 수 있음.

측정은 전부 아래 세 가지 규칙 위에서만 이루어짐. 개별 측정이 이 규칙을 다시 정하는
일은 없음.

- 좌표가 살아 있다는 것은 학습 구간 전체에서 최소 한 번이라도 켜졌다는 뜻임. 얼마나
  자주 켜졌는지로 거르는 기준은 어디에도 없음.
- 두 좌표가 함께 켜지는 정도는 부호를 살린 피어슨 상관계수임. 절댓값을 취하지 않음.
- 이미지 좌표와 문장 좌표의 짝짓기는 공동 활성 행렬 파일에 저장된 짝 하나를 그대로
  씀. 측정할 때마다 새로 짝을 찾지 않음.

두 가지 조건에서 같은 측정을 반복함.

- **coco_k8** 은 COCO 자료로 학습하고 입력 하나당 좌표 8개를 켜며 30번 반복 학습한
  조건임. 논문 그림 2의 조건과 같음.
- **cc3m_k32** 는 CC3M 자료로 학습하고 입력 하나당 좌표 32개를 켜며 10번 반복 학습한
  조건임. 논문 표 1의 조건과 같음.

---

## 2. 준비물

다섯 가지가 필요함. 하나라도 없으면 중간에 멈춤.

- **NVIDIA GPU 한 장, 메모리 10 GB 이상.** 학습과 벡터 추출이 모두 GPU에서 돌아감.
  메모리가 10 GB보다 작으면 7절의 메모리 부족 대처를 먼저 읽고 설정을 낮춰야 함.
- **Docker, 그리고 NVIDIA 컨테이너 실행 환경.** `docker run --gpus all` 이 동작해야
  함. 확인 명령은 `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
  이고, GPU 정보가 표로 출력되면 준비된 것임.
- **디스크 여유 공간 약 45 GB.** 내역은 다음과 같음. Hugging Face에서 받는 원본
  자료가 약 20 GB이고 대부분 COCO 사진임. 거기서 만든 벡터 표현 저장 파일이 COCO 약
  2 GB, CC3M 약 13 GB, ImageNet 약 3 GB임. COCO 객체 표기 파일이 241 MB임. 마지막으로
  결과물이 수 GB임. 공동 활성 행렬 파일 하나가 수백 MB라 결과물 쪽도 작지 않음.
- **Hugging Face 접근 토큰, 그리고 ILSVRC/imagenet-1k 자료 접근 권한.** ImageNet은
  승인이 필요한 자료임. 받는 방법은 이렇게 함. https://huggingface.co/datasets/ILSVRC/imagenet-1k
  에 로그인한 상태로 들어가서 이용 약관에 동의하고 접근을 신청함. 승인은 보통 바로
  나지만 하루 정도 걸릴 때도 있음. 승인된 뒤 https://huggingface.co/settings/tokens
  에서 읽기 권한 토큰을 하나 만들어 두면 됨.
- **네트워크.** 원본 자료 약 20 GB와 객체 표기 파일 241 MB를 내려받아야 함. CC3M은
  한 줄씩 흘려보내며 읽는 방식이라 실행 내내 네트워크를 씀.

---

## 3. 실행 명령

명령은 네 개임. 순서대로 그대로 실행하면 됨.

### 3.1 내려받기

```bash
git clone -b post-rebuttal https://github.com/<계정>/cross_modal_feature_heterogeneity.git
cd cross_modal_feature_heterogeneity
```

`<계정>` 자리에는 이 저장소를 받은 주소를 그대로 넣으면 됨. 받은 뒤 아래 명령으로
지금 가지가 맞는지 확인함. 출력이 `post-rebuttal` 이어야 함.

```bash
git branch --show-current
```

### 3.2 이미지 만들기

```bash
docker build -t vlm-sae .
```

10분에서 20분 정도 걸림. PyTorch와 CUDA가 포함된 공식 이미지를 기반으로 하고, 그
위에 이 저장소와 필요한 라이브러리를 설치함.

### 3.3 전체 실행

```bash
mkdir -p cache outputs
docker run --rm --gpus all \
  -e HF_TOKEN=여기에_토큰 \
  -e CONFIG=configs/post_rebuttal/clip_b32.yaml \
  -v "$PWD/cache:/app/repo/cache" \
  -v "$PWD/outputs:/app/repo/outputs" \
  vlm-sae
```

각 줄의 뜻은 다음과 같음.

- `--gpus all` 은 컨테이너가 GPU를 쓰게 함.
- `-e HF_TOKEN=여기에_토큰` 은 2절에서 만든 Hugging Face 토큰임. ImageNet 자료를
  받을 때만 쓰이지만, 없으면 해당 단계에서 멈추므로 처음부터 넣어두는 편이 나음.
- `-e CONFIG=configs/post_rebuttal/clip_b32.yaml` 은 어떤 실험을 돌릴지 지정함. 이
  값이 이 문서가 설명하는 실험임. 지정하지 않아도 같은 값이 기본으로 들어감.
- `-v "$PWD/cache:/app/repo/cache"` 는 내려받은 자료와 벡터 표현 저장 파일이 컨테이너
  밖에 남게 함. 이 연결이 없으면 컨테이너가 끝날 때 수십 GB가 사라지고 다시 받아야 함.
- `-v "$PWD/outputs:/app/repo/outputs"` 는 결과물이 컨테이너 밖에 남게 함.

터미널을 닫아도 계속 돌게 하려면 `--rm` 대신 `-d --name pr` 를 주고 실행한 뒤
`docker logs -f pr` 로 진행 상황을 봄.

### 3.4 한 단계만 실행하기

전체는 네 단계로 나뉘어 있고, `STAGE` 를 주면 그중 하나만 돌릴 수 있음.

| STAGE 값 | 하는 일 |
|---|---|
| `figure2` | COCO 벡터 추출, COCO 모델 학습, 공동 활성 행렬 계산, 그림 2 그리기 |
| `table1` | CC3M 벡터 추출, 네 가지 방법 학습, 평가, 표 1 쓰기 |
| `rebuttal` | 두 번째 COCO 모델 학습, 추가 공동 활성 행렬 계산, 후속 측정 실행 |
| `report` | 지금까지 나온 결과를 한 문서로 묶기 |
| `all` | 위 네 가지를 이 순서대로 전부 실행함. 기본값임 |

예를 들어 후속 측정만 다시 돌리려면 이렇게 함.

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=여기에_토큰 \
  -e CONFIG=configs/post_rebuttal/clip_b32.yaml \
  -e STAGE=rebuttal \
  -v "$PWD/cache:/app/repo/cache" \
  -v "$PWD/outputs:/app/repo/outputs" \
  vlm-sae
```

### 3.5 중단된 뒤 다시 시작하기

3.3의 명령을 그대로 다시 실행하면 됨. 모든 단계가 이미 만들어진 결과를 보고 건너뛰게
되어 있어서, 처음부터 다시 하지 않음. 구체적으로 다음을 건너뜀.

- 벡터 표현 저장 파일이 완성되어 있으면 해당 자료의 추출을 통째로 건너뜀. CC3M처럼
  아직 완성되지 않은 경우에는 이미 기록한 줄 수만큼 건너뛰고 이어서 받음.
- 학습된 모델 파일이 있으면 그 모델의 학습을 건너뜀.
- 공동 활성 행렬 파일이 있으면 다시 계산하지 않음. 다만 그 파일이 지금 필요한 조건과
  다른 조건에서 만들어진 것이면 다시 계산함. 이 판단은 파일 옆의 설명 파일을 읽어서
  자동으로 함.
- 후속 측정은 `<이름>.json` 이 있으면 건너뜀. 다시 돌리고 싶은 측정이 있으면 그
  `.json` 파일을 지우고 3.4의 `STAGE=rebuttal` 을 실행하면 그 측정만 다시 함.

---

## 4. 어떤 순서로 얼마나 걸리는가

아래 시간은 GPU 한 장 기준 어림값임. 네트워크 속도와 디스크 속도에 따라 두 배까지
차이 날 수 있음. 전체는 대략 12–24시간으로 보면 됨.

| 순서 | 하는 일 | 어림 시간 |
|---|---|---|
| 1 | COCO 사진과 문장을 내려받아 벡터로 바꿔 `cache/clip_b32_coco/` 에 저장함. 약 20 GB를 받는 구간이라 네트워크가 가장 크게 영향을 줌 | 약 1–2시간 |
| 2 | COCO 벡터로 희소 오토인코더를 30번 반복 학습함 | 약 20–40분 |
| 3 | 학습 구간 전체에 대해 공동 활성 행렬을 계산해 저장함 | 약 10–20분 |
| 4 | 그림 2를 그리고 구간별 수치 표를 씀 | 1분 미만 |
| 5 | CC3M 287만 쌍을 흘려보내며 벡터로 바꿔 `cache/clip_b32_cc3m/` 에 저장함. 전체에서 가장 긴 구간임 | 약 6–12시간 |
| 6 | CC3M 벡터로 네 가지 방법을 각각 세 개의 난수 씨앗값으로 10번씩 반복 학습함. 모델 12개를 만드는 셈임 | 약 3–6시간 |
| 7 | ImageNet 검증용 사진 5만 장과 분류 문장을 벡터로 바꿔 `cache/clip_b32_imagenet/` 에 저장함 | 약 20–40분 |
| 8 | 복원 오차, 검색 정확도, 무학습 분류 정확도를 방법별로 재고 표 1을 씀 | 약 10–20분 |
| 9 | 두 번째 COCO 모델을 학습하고, 추가 공동 활성 행렬 네 개를 조건마다 계산함. CC3M 쪽은 287만 쌍을 여러 번 훑어야 해서 시간이 걸림 | 약 1–2시간 |
| 10 | COCO 객체 표기 파일 241 MB를 받고, 후속 측정을 조건마다 실행함. COCO 조건에서 8가지, CC3M 조건에서 11가지임 | 약 1–3시간 |
| 11 | 전체를 한 문서로 묶음 | 1분 미만 |

---

## 5. 보내줄 파일

실행이 끝나면 아래 명령 하나로 보낼 파일을 한 덩어리로 묶을 수 있음. 저장소 최상위
폴더에서 실행하면 됨.

```bash
bash scripts/collect_deliverables.sh
```

만들어지는 파일은 `outputs/post_rebuttal_deliverables.tar.gz` 하나임. 이 파일만
보내주면 됨. 크기는 수십 MB 정도임.

안에 들어가는 것은 다음과 같음.

- 모든 보고서 `.md` 파일. 전체를 묶은 `post_rebuttal_results.md`, 표 1, 그림 2의 구간별
  수치 표, 후속 측정 11가지의 보고서가 모두 들어감.
- 모든 그림 파일. `.pdf` 와 같은 내용의 `.png` 가 함께 들어감.
- 모든 수치 파일 `.json`. 보고서에 실린 숫자가 전부 여기에도 들어 있어서, 표를 다시
  만들거나 다른 방식으로 그리고 싶을 때 쓸 수 있음.
- 표 1의 LaTeX 판 `.tex`.

들어가지 않는 것은 두 가지임. 학습된 모델 파일과 공동 활성 행렬 `.npz` 파일임. 둘 다
결과가 아니라 결과를 만들기 위한 중간물이고, 공동 활성 행렬은 하나가 수백 MB라
보내기 어려운 크기가 됨.

---

## 6. 진행 상황 보기

기록은 화면에 그대로 나옴. 컨테이너를 배경으로 돌렸다면 `docker logs -f pr` 로 봄.
줄 앞머리를 보면 지금 어디인지 알 수 있음.

- `[extract]` 로 시작하면 벡터 표현을 만드는 중임. 처리 속도와 남은 시간이 30초마다
  한 줄씩 나옴.
- `[train]` 으로 시작하면 학습 중임.
- `[panel]` 또는 `[multi_density] saved` 가 보이면 공동 활성 행렬을 계산해 저장한
  것임.
- `[post_rebuttal] <조건>: start <측정 이름>` 은 후속 측정 하나가 시작된 것이고,
  `end <측정 이름>, <숫자> s elapsed` 는 그 측정이 끝나고 몇 초 걸렸는지 알려주는
  것임.
- `[post_rebuttal][skip]` 은 이미 만들어져 있어서 건너뛴 것임. 다시 실행할 때 대부분의
  줄이 이것이면 정상임.
- `[done] post_rebuttal -> outputs/post_rebuttal/post_rebuttal_results.md` 가 마지막
  줄임.

---

## 7. 오류가 났을 때

### 7.1 측정 하나가 실패했을 때

측정 하나가 실패해도 나머지는 계속 돌아감. 실패한 측정의 오류 내용은 파일로 남음.
위치는 `outputs/post_rebuttal/rebuttal/coco_k8/<측정 이름>.error.txt` 또는
`outputs/post_rebuttal/rebuttal/cc3m_k32/<측정 이름>.error.txt` 임. 모두 끝난 뒤
프로그램은 0이 아닌 값으로 종료하고, 마지막 줄에 실패한 측정 이름을 전부 적어줌.
묶음 문서에도 그 자리에 "Missing" 이라고 표시되므로, 빠진 측정이 있었는지 문서만
봐도 알 수 있음. 실패 파일을 그대로 보내주면 원인을 확인할 수 있음.

### 7.2 GPU 메모리가 부족할 때

`CUDA out of memory` 라는 문구가 나오면 한 번에 처리하는 표본 수를 줄이면 됨. 고칠
값은 두 군데임.

- 학습에서 부족한 경우에는 `configs/post_rebuttal/clip_b32_coco.yaml` 과
  `configs/post_rebuttal/clip_b32_cc3m.yaml` 의 `training.batch_size` 를 1024에서
  512나 256으로 낮춤. 두 파일 모두 저장소 최상위 폴더 아래에 있음.
- 공동 활성 행렬 계산에서 부족한 경우에도 같은 `training.batch_size` 를 낮추면 됨. 그
  계산이 같은 값을 읽어서 씀.

값을 낮추면 학습 결과가 조금 달라질 수 있지만, 이 측정들이 보는 것은 개별 수치의
소수점 아래가 아니라 조건 사이의 차이라서 결론에는 영향이 없음. 값을 고친 뒤에는
이미 학습된 모델 파일을 지우고 다시 실행해야 새 값이 반영됨.

### 7.3 Hugging Face 토큰 오류

`401` 또는 `403`, 혹은 `gated` 라는 문구가 나오면 토큰 문제임. 확인할 것은 세 가지임.

- `-e HF_TOKEN=...` 을 실제로 넘겼는지 확인함. 넘기지 않으면 ImageNet 단계에서 멈춤.
- 2절 주소에서 ILSVRC/imagenet-1k 접근 신청이 승인되었는지 확인함. 신청만 하고 승인
  전이면 토큰이 있어도 거부됨.
- 토큰이 만료되지 않았는지, 읽기 권한이 있는지 확인함.

ImageNet은 표 1의 무학습 분류 정확도 열에만 쓰임. 급하면 먼저 `STAGE=figure2` 와
`STAGE=rebuttal` 을 돌려두고, 토큰이 준비된 뒤 `STAGE=all` 을 다시 실행하면 됨.

### 7.4 저장 파일이 중간까지만 만들어졌을 때

벡터 표현을 만드는 중에 멈췄다면 그대로 다시 실행하면 됨. 만들다 만 조각 파일이
`cache/<이름>/parts/` 에 남고 진행 상황이 함께 기록되어 있어서, 이미 처리한 만큼은
건너뛰고 이어서 함.

다시 실행해도 같은 자리에서 계속 실패한다면 그 자료의 저장 폴더를 통째로 지우고
처음부터 받는 것이 확실함. 지울 폴더는 `cache/clip_b32_coco/`,
`cache/clip_b32_cc3m/`, `cache/clip_b32_imagenet/` 중 실패한 것 하나임.
`cache/hf/` 는 원본 자료를 받아둔 곳이라 지우지 않는 편이 나음. 지우면 20 GB를 다시
받게 됨.

COCO 객체 표기 파일이 중간까지만 받아졌다면 `cache/coco_annotations/` 안에
`.part` 로 끝나는 파일이 남아 있음. 다시 실행하면 그 지점부터 이어서 받으므로 따로
할 일은 없음.

---

## 8. 설정값 표

고칠 일이 있을 때만 보면 됨. 기본값 그대로 두는 것이 정상임.

### 8.1 `configs/post_rebuttal/clip_b32.yaml`

| 설정 이름 | 뜻 | 기본값 |
|---|---|---|
| `kind` | 어떤 실행 절차를 쓸지 정함. 이 값이 `post_rebuttal` 이어야 네 단계 실행이 됨 | `post_rebuttal` |
| `figure2` | 그림 2를 만드는 설정 파일을 가리킴 | `clip_b32_coco.yaml` |
| `table1` | 표 1을 만드는 설정 파일을 가리킴 | `clip_b32_cc3m.yaml` |
| `rebuttal.tau` | 두 좌표가 함께 켜지는 상관계수가 이 값을 넘어야 그 둘을 한 짝으로 보고 방향 차이를 잼 | `0.4` |
| `rebuttal.null_seed` | 사진과 문장의 짝을 일부러 뒤섞어 잡음 수준을 재는 데 쓰는 난수 씨앗값. 0은 "섞지 않음"을 뜻하므로 0이면 안 됨 | `7` |
| `rebuttal.n_boot` | 신뢰구간 하나를 만들 때 자료를 중복 허용으로 다시 뽑는 횟수. 클수록 구간이 안정되고 느려짐 | `1000` |
| `rebuttal.settings` | 어떤 조건을 분석할지 고름. `coco_k8` 은 논문 그림 2 조건, `cc3m_k32` 는 논문 표 1 조건임 | `[coco_k8, cc3m_k32]` |
| `rebuttal.analyses` | 어떤 측정을 돌릴지 고름. `all` 이면 등록된 11가지를 전부 돌림. 측정 이름을 직접 적을 수도 있음 | `[all]` |
| `rebuttal.coco_seed_b` | 두 번째 COCO 모델의 난수 씨앗값. 같은 설정으로 두 번 학습했을 때 얼마나 달라지는지 재려면 첫 번째와 달라야 함 | `1` |
| `output.root` | 결과물을 쓸 폴더 | `outputs/post_rebuttal` |

### 8.2 `configs/post_rebuttal/clip_b32_coco.yaml` 과 `clip_b32_cc3m.yaml`

두 파일이 같은 이름의 설정을 쓰고 값만 다름. 아래 표에서 앞의 값이 COCO 쪽, 뒤의
값이 CC3M 쪽임.

| 설정 이름 | 뜻 | 기본값 |
|---|---|---|
| `training.k` | 입력 하나당 켜지는 좌표의 개수 | COCO `8`, CC3M `32` |
| `training.latent_size` | 좌표의 총 개수. 이미지 쪽과 문장 쪽이 절반씩 나눠 가지므로 8192이면 한쪽당 4096임 | `8192` |
| `training.num_epochs` | 학습 자료 전체를 몇 번 반복해서 볼지 | COCO `30`, CC3M `10` |
| `training.batch_size` | 한 번에 처리하는 입력 쌍의 수. GPU 메모리가 부족할 때 낮추는 값임 | `1024` |
| `training.lr` | 학습이 한 걸음에 가중치를 얼마나 바꿀지 정하는 값 | `0.0005` |
| `training.weight_decay` | 가중치가 지나치게 커지지 않도록 누르는 정도 | `0.00001` |
| `training.max_grad_norm` | 한 걸음의 변화량이 이 값을 넘으면 잘라냄. 학습이 튀는 것을 막음 | `1.0` |
| `training.warmup_ratio` | 처음 몇 퍼센트 구간 동안 학습 속도를 천천히 올릴지 | `0.05` |
| `training.device` | 계산을 어디서 할지. `cuda` 는 GPU, `cpu` 는 중앙처리장치임 | `cuda` |
| `training.seed` | 목록을 따로 주지 않을 때 쓰는 난수 씨앗값 | `0` |
| `training.seeds` | 난수 씨앗값 목록. 목록에 적힌 수만큼 모델을 따로 학습함 | COCO `[0]`, CC3M `[0, 1, 2]` |
| `kind` | 어떤 실행 절차를 쓸지 정함 | COCO `multi_density`, CC3M `cc3m_downstream` |
| `models` 또는 `model` | 벡터 표현을 만들 시각 언어 모델의 설정 파일을 가리킴. COCO 쪽은 여러 개를 적을 수 있어 목록이고, CC3M 쪽은 하나임 | `../models/clip_b32.yaml` |
| `methods` | CC3M 쪽에서 학습하고 비교할 방법 목록을 가리킴. 공유 사전, 모달리티별 사전, 두 가지 보조 손실, 그리고 사후 정렬까지 다섯 가지임 | `../cc3m/_shared.yaml#methods` |
| `cache.cache_dir` | 벡터 표현을 저장할 폴더 | COCO `cache/clip_b32_coco`, CC3M `cache/clip_b32_cc3m` |
| `cache.dataset` | 어떤 자료를 쓸지 | COCO `coco`, CC3M `cc3m` |
| `cache.split` | 자료의 어느 구간으로 학습할지 | `train` |
| `eval.recon` | 복원 오차를 잴지 여부. CC3M 쪽에만 있음 | `true` |
| `eval.retrieval` | 이미지와 문장 사이 검색 정확도를 잴지 여부. CC3M 쪽에만 있음 | `true` |
| `eval.zeroshot` | ImageNet 무학습 분류 정확도를 잴지 여부. CC3M 쪽에만 있음 | `true` |
| `eval.zeroshot_variant` | 무학습 분류를 어떤 방식으로 잴지. `raw` 는 좌표를 하나도 걸러내지 않는 방식임 | `raw` |
| `eval.max_fire_rate` | 너무 자주 켜지는 좌표를 분류에서 뺄 때 쓰는 기준. `raw` 방식에서는 좌표를 빼지 않으므로 쓰이지 않음 | `0.5` |
| `eval.recon_template_seed` | 복원 오차를 잴 때 분류 문장을 고르는 난수 씨앗값 | `0` |
| `output.root` | 결과물을 쓸 폴더 | COCO `outputs/post_rebuttal/coco_clip_b32`, CC3M `outputs/post_rebuttal/cc3m_clip_b32` |
| `output.save_decoders` | 좌표 방향 행렬을 따로 저장할지 여부. 합성 자료 실험에서만 쓰이고 이 실행에서는 영향이 없음 | `true` |
