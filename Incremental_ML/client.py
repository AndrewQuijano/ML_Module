import socket
from sys import argv, exit


# Will send (x, y) to cluster
def main():
    if len(argv) != 3:
        print("Usage: python3 client.py <IP Address> <Port Number>")
        exit(0)

    # Test Values for connection
    ip = argv[1]
    port = int(argv[2])


    # Keep sending (x, y) coordinates...
    while True:
        try:
            var = input("ML> ")

            if var == "exit":
                break
            else:
                # connect to the server on local computer
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((ip, port))

                client_socket.send(var.encode())
                y_hat = client_socket.recv(1024).decode()
                print(y_hat)

        except EOFError:
            print("EOF detected, exiting!")
            break

        except socket.timeout:
            print("Timeout Exception!")
            continue

        except KeyboardInterrupt:
            print('CTRL-C received, exiting!')
            break


if __name__ == "__main__":
    main()
