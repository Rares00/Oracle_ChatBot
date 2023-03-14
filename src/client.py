import socket


s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
s.connect(('127.0.0.1', 9000))
print("Connect success!")

def make_sock():
    send_data=input()
    s.send(send_data.encode("utf-8"))
    ##print("已发送!")
    return None


def recv_scok():
    get_data = s.recv(1024)
    get_data.decode()
    datastring = str(get_data, 'utf8')
    if get_data is None:
        print("No feedback received")
    else:
        print("Weather bot:")

        print(datastring)
    return None


if __name__ == "__main__":
    flag = 1
    while True:
        if flag == 0:
            s.close()
            break
        else:
            print("What you want to ask:")
            make_sock()
        # 接收消息
        recv_scok()
        print("Press any key continue of 0 exit")
        a = input()
        flag = a