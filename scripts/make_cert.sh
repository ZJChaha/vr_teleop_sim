#!/usr/bin/env bash
# ============================================================================
# 重签自签 HTTPS 证书，SAN 覆盖**当前**本机 IP
#
#      cd ~/vr_teleop_gripper
#      ./tools/make_cert.sh
#
# ★ 什么时候要跑：换了 WiFi / 换了网段，本机 IP 变了。
#   （server.py 绑 0.0.0.0、index.html 用 location.host，都**不用改**；
#     只有这张自签证书把 IP 写进了 CN/SAN。）
# ★ 旧证书/私钥会先备份到 .cert_backup/（或 VR_TELEOP_BACKUP_DIR），可回滚。
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

# ---- 收集当前 IP ----
# 也可额外传参把别的 IP/域名也写进 SAN（换网来回切时很有用）：
#     ./scripts/make_cert.sh 192.168.1.50 10.0.0.7
IPS="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
[ -z "$IPS" ] && { echo "[make_cert] 找不到全局 IPv4（连着网吗？）"; exit 1; }
PRIMARY="$(echo "$IPS" | head -1)"

SAN="DNS:localhost,DNS:$(hostname),DNS:$(hostname).local,IP:127.0.0.1"
for ip in $IPS; do SAN="$SAN,IP:$ip"; done
# 额外参数：像 IPv4 就当 IP，否则当 DNS
for extra in "$@"; do
    case "$extra" in
        *[!0-9.]*|"") SAN="$SAN,DNS:$extra" ;;
        *)            SAN="$SAN,IP:$extra" ;;
    esac
done

echo "============================================================"
echo " 重签证书"
echo "------------------------------------------------------------"
echo " 主 IP  : $PRIMARY"
echo " 全部 IP: $(echo "$IPS" | tr '\n' ' ')"
echo " SAN    : $SAN"
echo "============================================================"

# ---- 备份 ----（只增不删）
TS="$(date +%Y%m%d_%H%M%S)"
BK="${VR_TELEOP_BACKUP_DIR:-$ROOT/.cert_backup}"
if [ -f "$ROOT/cert.pem" ]; then
    mkdir -p "$BK"
    cp -p "$ROOT/cert.pem" "$BK/cert_before_$TS.pem"
    cp -p "$ROOT/key.pem"  "$BK/key_before_$TS.pem"
    echo " 旧证书已备份: $BK/cert_before_$TS.pem"
    echo "              $BK/key_before_$TS.pem"
fi

# ---- 生成 ----（自签；浏览器仍会提示"不受信任"，点继续即可）
# ★ 只加 subjectAltName，其余走 openssl 默认 —— 与改造前那张证书的属性**逐项一致**
#   （openssl 默认给的是 basicConstraints=critical,CA:TRUE + SKI/AKI，正是原来那张）。
#   踩过的坑：① 自己再 -addext basicConstraints 会**重复该扩展**；② 写成 CA:FALSE
#   会让自签证书当不了信任锚（verify 报 "unable to get local issuer certificate"）。
#   本次唯一的实质改动 = 加 SAN（现代浏览器只认 SAN，不认 CN）。
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 3650 \
    -keyout "$ROOT/key.pem" -out "$ROOT/cert.pem" \
    -subj "/CN=$PRIMARY" \
    -addext "subjectAltName=$SAN" >/dev/null 2>&1 || {
        echo "[make_cert] ✗ openssl 生成失败（openssl 需 >= 1.1.1 支持 -addext）"; exit 1; }
chmod 600 "$ROOT/key.pem"
chmod 644 "$ROOT/cert.pem"

echo
echo " 新证书:"
openssl x509 -in "$ROOT/cert.pem" -noout -subject -dates | sed 's/^/   /'
echo -n "   SAN: "
openssl x509 -in "$ROOT/cert.pem" -noout -ext subjectAltName 2>/dev/null | tail -1 | tr -d ' '
echo
echo " 头显浏览器地址（任选一个本机 IP，头显与电脑同网段即可）:"
for ip in $IPS; do echo "   https://$ip:8444"; done
echo " （改完证书要**重启** ./run.sh 才生效）"
