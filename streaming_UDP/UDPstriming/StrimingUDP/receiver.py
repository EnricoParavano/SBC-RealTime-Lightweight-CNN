import socket
import struct
import cv2
import numpy as np
from collections import defaultdict

UDP_IP = "0.0.0.0"      # ascolta su tutte le interfacce
UDP_PORT = 5200         # deve corrispondere a quello usato dal sender
BUFFER_SIZE = 65536     # UDP max packet size

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(5)

print(f"Receiver in ascolto su {UDP_IP}:{UDP_PORT}...")

frames = defaultdict(lambda: {"chunks": {}, "total": 0})

while True:
    try:
        packet, addr = sock.recvfrom(BUFFER_SIZE)

        # Header: frame_id (4) + total_chunks (4) + chunk_index (4) + chunk_size (4)
        header = packet[:16]
        frame_id, total_chunks, chunk_index, chunk_size = struct.unpack('!IIII', header)
        chunk_data = packet[16:]

        if len(chunk_data) != chunk_size:
            print("Pacchetto corrotto o incompleto, scartato.")
            continue

        frame = frames[frame_id]
        frame["chunks"][chunk_index] = chunk_data
        frame["total"] = total_chunks

        # Se tutti i chunk sono arrivati
        if len(frame["chunks"]) == total_chunks:
            # Ricostruzione frame
            ordered_chunks = [frame["chunks"][i] for i in range(total_chunks)]
            frame_bytes = b"".join(ordered_chunks)
            img_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame_image is not None:
                cv2.imshow("Ricevuto via UDP", frame_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Rimuove il frame completato
            del frames[frame_id]

    except socket.timeout:
        print("Timeout UDP. Nessun pacchetto ricevuto di recente.")
    except KeyboardInterrupt:
        print("\nRicezione interrotta.")
        break

sock.close()
cv2.destroyAllWindows()