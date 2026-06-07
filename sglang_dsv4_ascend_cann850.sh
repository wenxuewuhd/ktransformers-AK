#!/usr/bin/env bash
set -euo pipefail

# ===== 用户配置 =====
export IMAGE=${IMAGE:-lmsysorg/sglang:deepseek-v4-npu-910b}
export NAME=${NAME:-sglang_ds_v4}
export PORT_RANGE=${PORT_RANGE:-8000-8010}
export HOST_PORT_RANGE=${HOST_PORT_RANGE:-8070-8080}
export SHM_SIZE=${SHM_SIZE:-16g}
export CACHE_DIR=${CACHE_DIR:-$HOME/.cache}
export MODEL_DIR=${MODEL_DIR:-/home/y00359136/models}
export SGLANG_DIR=${SGLANG_DIR:-/home/y00359136/code}
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
VISIBLE_IDS=$(printf "%s\n" "${DLIST[@]}" | sed -E 's#.*/davinci([0-9]+)#\1#' | paste -sd, -)

#for d in  /dev/davinci1 /dev/davinci4 /dev/davinci5 /dev/davinci7; do
#  [[ -e "$d" ]] && DEVICES+=(--device "$d")
#done

MOUNTS=(
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /var/log/npu/:/usr/slog
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro
  -v /usr/local/dcmi:/usr/local/dcmi
  -v "${MODEL_DIR}":/workspace/models/
  -v "${SGLANG_DIR}":/workspace/code/
  -v "${CACHE_DIR}":/root/.cache
)

# 组装容器内部初始化命令（注意必须是一行）
CMD='

# 0. 强制让 APT 使用环境变量中的代理
echo "==> 配置 APT 代理..."
mkdir -p /etc/apt/apt.conf.d
echo "Acquire::http::Proxy \"$http_proxy\";" > /etc/apt/apt.conf.d/proxy.conf
echo "Acquire::https::Proxy \"$https_proxy\";" >> /etc/apt/apt.conf.d/proxy.conf

set -e
# 1. 配置 APT 镜像源 (使用华为云公共镜像源 repo.huaweicloud.com)
echo "==> 配置 APT 镜像源..."
sed -i "s@http://ports.ubuntu.com/ubuntu-ports/@http://repo.huaweicloud.com/ubuntu-ports/@g" /etc/apt/sources.list
# 如果基础镜像中包含原有的 mirrors.tools.huawei.com，也一并替换掉
sed -i "s@http://mirrors.tools.huawei.com/ubuntu-ports/@http://repo.huaweicloud.com/ubuntu-ports/@g" /etc/apt/sources.list
sed -i "s@https://mirrors.tools.huawei.com/ubuntu-ports/@http://repo.huaweicloud.com/ubuntu-ports/@g" /etc/apt/sources.list

# 2. 安装并配置 SSH 服务
echo "==> 安装 SSH 服务..."
apt-get update
apt-get install -y openssh-server
apt-get install -y libgl1-mesa-glx

echo "==> 配置 SSH 参数 (端口 8010)..."
mkdir -p /var/run/sshd
# 修改配置文件：允许 Root 登录、开启密码认证、修改端口为 8010
sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config
sed -i "s/#PasswordAuthentication yes/PasswordAuthentication yes/" /etc/ssh/sshd_config
sed -i "s/#Port 22/Port 8010/" /etc/ssh/sshd_config
# 确保端口确实被修改（处理没有默认注释行的情况）
echo "Port 8010" >> /etc/ssh/sshd_config

echo "==> 设置 Root 密码为 123456..."
echo "root:123456" | chpasswd

echo "==> 启动 SSH 服务..."
service ssh start

# 3. 配置 Pip 镜像源 (使用华为云公共 Pip 源)
echo "==> 配置 Pip 镜像源..."
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.trusted-host repo.huaweicloud.com

echo "==> 升级 pip 和基础工具..."
pip install --upgrade pip
pip install tabulate rich

# 4. 验证与显示
echo "==> 可见 NPU: $NPU_VISIBLE_DEVICES"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
fi

echo "容器初始化完毕，已进入交互模式。"
exec bash
'

docker run --rm \
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
  -e NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"\
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --privileged=true \
  --shm-size "${SHM_SIZE}" \
  -p "${HOST_PORT_RANGE}:${PORT_RANGE}" \
  -it "${IMAGE}" /bin/bash -c "${CMD}"
