#!/usr/bin/env bash
set -euo pipefail

# ===== 用户配置 =====
export IMAGE=${IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts}
export NAME=${NAME:-ktransformers}
export PORT_RANGE=${PORT_RANGE:-8000-8010}
export HOST_PORT_RANGE=${HOST_PORT_RANGE:-18070-18080}
export SHM_SIZE=${SHM_SIZE:-32g}
export PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-auto}
# =====================

DEVICES=()
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "$d" ]] && DEVICES+=(--device "$d")
done

if [[ "$NPU_VISIBLE_DEVICES" == "auto" ]]; then
  mapfile -t DLIST < <(ls -1 /dev/davinci[0-9]* 2>/dev/null | sort -V)
else
  IFS=',' read -ra IDS <<<"$NPU_VISIBLE_DEVICES"
  DLIST=()
  for i in "${IDS[@]}"; do
    [[ -e "/dev/davinci${i}" ]] && DLIST+=("/dev/davinci${i}")
  done
fi
for d in "${DLIST[@]}"; do DEVICES+=(--device "$d"); done

MOUNTS=(
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /var/log/npu/:/usr/slog
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro
  -v /usr/local/dcmi:/usr/local/dcmi
  -v "${PROJECT_DIR}":/workspace
)

CMD='
set -e

echo "==> 配置 Pip 镜像源..."
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.trusted-host repo.huaweicloud.com

echo "==> 全局禁用 Python SSL 校验（内网代理兼容）..."
SITEPKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "${SITEPKG}/sitecustomize_ssl.py" <<PYEOF
import ssl, os, urllib3
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
_orig = requests.Session.send
def _patched(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig(self, request, **kwargs)
requests.Session.send = _patched
PYEOF
EXISTING=$(cat "${SITEPKG}/sitecustomize.py" 2>/dev/null || true)
if ! echo "$EXISTING" | grep -q "sitecustomize_ssl"; then
  echo "import sitecustomize_ssl" >> "${SITEPKG}/sitecustomize.py"
fi

echo "==> 当前代理:"
env | grep -i proxy || true
echo "==> pip 配置:"
pip config list || true

echo "容器初始化完毕，已进入交互模式。"
exec bash
'

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -it -d \
  --name "${NAME}" \
  "${DEVICES[@]}" \
  "${MOUNTS[@]}" \
  -e http_proxy="${http_proxy:-}" \
  -e https_proxy="${https_proxy:-}" \
  -e HTTP_PROXY="${http_proxy:-}" \
  -e HTTPS_PROXY="${https_proxy:-}" \
  -e no_proxy="${no_proxy:-}" \
  -e NO_PROXY="${no_proxy:-}" \
  -e ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  -e NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --privileged=true \
  --shm-size "${SHM_SIZE}" \
  -w /workspace \
  -p "${HOST_PORT_RANGE}:${PORT_RANGE}" \
  "${IMAGE}" /bin/bash -c "${CMD}"

echo "[INFO] Started container: ${NAME}"
echo "[INFO] Port mapping: ${HOST_PORT_RANGE} -> ${PORT_RANGE}"
echo "[INFO] Enter container: docker exec -it ${NAME} bash"
