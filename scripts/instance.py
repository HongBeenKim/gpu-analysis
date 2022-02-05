from openpyxl import Workbook
import subprocess
import math
import time

wb = Workbook()
ws = wb.active

for mat_size_in_kb in range(100, 5001, 100):
    N = int(16 * math.sqrt(mat_size_in_kb))

    sum_ = 0
    for i in range(10):
        env = {'CUDA_VISIBLE_DEVICES': 'MIG-6e5ecf1c-980b-53b4-b79e-df70177fd284',
               'CUDA_MPS_PIPE_DIRECTORY': '/tmp/MIG-6e5ecf1c-980b-53b4-b79e-df70177fd284'}
        disturbing_proc = subprocess.Popen(['./single', str(N), '0'], env=env)

        env = {'CUDA_VISIBLE_DEVICES': 'MIG-3234bc3b-83f3-5e3a-940e-d1c72da74e00',
               'CUDA_MPS_PIPE_DIRECTORY': '/tmp/MIG-3234bc3b-83f3-5e3a-940e-d1c72da74e00'}
        proc = subprocess.Popen(['./single', str(N), '1'], env=env, stdout=subprocess.PIPE)
        out, err = proc.communicate()
        out = float(out.decode('utf-8').strip())
        sum_ += out

        print(out)
        disturbing_proc.wait()
        time.sleep(1)

    row = [N, N * N / 256, sum_ / 10]
    print(row)
    ws.append(row)

wb.save('result_instance.xlsx')

