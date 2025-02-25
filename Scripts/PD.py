import paramiko
import signal
import sys

"""
このスクリプトはUnitree A1にSSH接続してPD制御を実行するためのスクリプトです。
"""

# 接続情報
host = "192.168.12.1"
port = 22  # SSHポート（デフォルト22）
username = "pi"
password = "123"  # 鍵認証なら省略

# 実行するスクリプトのパス
remote_script_path = "/home/pi/25demo.sh"

# SSH クライアントの作成
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

def signal_handler(sig, frame):
    print("Ctrl+C detected, terminating remote script...")
    ssh.exec_command(f"pkill -f {remote_script_path}")
    ssh.close()
    sys.exit(0)

# Set up the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

try:
    # 接続
    ssh.connect(host, port, username, password)

    # シェルスクリプトを実行 with a pseudo-terminal
    transport = ssh.get_transport()
    channel = transport.open_session()
    channel.get_pty()  # 疑似端末を要求
    channel.exec_command(f"bash {remote_script_path}")

    # 結果を取得
    while True:
        if channel.recv_ready():
            print("標準出力:", channel.recv(1024).decode())
        if channel.recv_stderr_ready():
            print("エラー出力:", channel.recv_stderr(1024).decode())
        if channel.exit_status_ready():
            break

finally:
    ssh.close()
