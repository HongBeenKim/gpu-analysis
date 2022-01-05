from openpyxl import Workbook
import subprocess
import math

wb = Workbook()
ws = wb.active

for mat_size_in_kb in range(100, 5001, 100):
    N = int(16 * math.sqrt(mat_size_in_kb))
    cmd = ['./mps', str(N), '1']
    fd_popen = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout
    elap_single = float(str(fd_popen.read().strip(), 'utf-8'))
    fd_popen.close()

    subprocess.Popen(['./mps', str(N), '0'])
    fd_popen = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout
    elap_dual = float(str(fd_popen.read().strip(), 'utf-8'))
    fd_popen.close()

    row = [N, N * N / 256, elap_dual / elap_single]
    ws.append(row)

wb.save('result_mps.xlsx')

