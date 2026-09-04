#!/usr/bin/env python
# -- coding: utf-8 --
"""การประมาณค่าพายด้วยวิธี Monte Carlo
"""
import random
import math
import os 

import matplotlib.pyplot as plt

num_samples = 50001 # จำนวนตัวอย่าง
r = 5.0 # รัศมีวงกลมหรือครึ่งหนึ่งของความยาวด้านของสี่เหลี่ยมจัตุรัส
a, b = 0, r # ช่วง [a, b] คือ [0, r]

in_circle = 0
fig, ax = plt.subplots()
ax.add_patch(plt.Circle((0, 0), r, color='g'))
ax.set_xlim([0, r])
ax.set_ylim([0, r])
exp = 0

for i in range(num_samples): # สุ่มไปเรื่อยๆ
    x = random.uniform(a, b)
    y = random.uniform(a, b)
    
    if x**2 + y**2 <= r**2: # หากอยู่ในวงกลมก็นับ
        in_circle += 1
        plt.plot(x, y, 'ro', markersize=3)
    else:
        plt.plot(x, y, 'bo', markersize=3)

    if i > 0 and i % 5000 == 0:
        print('[Result] pi estimation: {0:.5f}'.format(4*float(in_circle)/i)) # แสดงค่าพายที่ประมาณจากการทดลอง
        pi_est = 4*float(in_circle)/i
        plt.title("pi compute: {0:.5f} num_samples: {1:5d}".format(4*float(in_circle)/i, i))
        plt.savefig('{}.jpg'.format(exp))
        # plt.draw()
        # plt.waitforbuttonpress(0) # รอๆๆๆๆ พิมพ์หรือคลิ๊ก
        # plt.close()
        exp += 1

os.system('ffmpeg -framerate 10 -i \%d.jpg -loop 0 pi_animation.gif')