from openpyxl import Workbook
import subprocess
import math

wb = Workbook()
ws = wb.active

for mat_size_in_kb in range(100, 5001, 100):
    N = int(16 * math.sqrt(mat_size_in_kb))
    cmd = ['./cache', str(N)]
    fd_popen = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout
    data = str(fd_popen.read().strip(), 'utf-8')
    fd_popen.close()

    row = [N * N / 256, data]
    ws.append(row)

wb.save('result.xlsx')

