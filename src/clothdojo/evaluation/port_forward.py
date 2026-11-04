import socket
import sys
import threading
def forward(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass
        try:
            dst.close()
        except Exception:
            pass
def main():
    l_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8733
    t_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8731
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', l_port))
    srv.listen(32)
    print(f'FORWARD_READY: 0.0.0.0:{l_port} -> 127.0.0.1:{t_port}', flush=True)
    while True:
        client, _ = srv.accept()
        tgt = socket.socket()
        try:
            tgt.connect(('127.0.0.1', t_port))
        except Exception:
            client.close()
            continue
        threading.Thread(target=forward, args=(client, tgt), daemon=True).start()
        threading.Thread(target=forward, args=(tgt, client), daemon=True).start()
if __name__ == '__main__':
    main()
