#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:04 2019

@author: yujia
"""

import matplotlib.pyplot as plt
import numpy as np

offset = 0.6

nll = [-0.8414, -0.8783, -0.9314,  -0.9598, -1.5532, -0.9624] #linkedin
#nll = [-0.8665, -0.7938, -offset,  -1.3648, -0.9818, -1.1117]
x = np.arange(len(nll))

ll = -np.asarray(nll) -offset

#width = 0.35  # the width of the bars
#
#fig, ax = plt.subplots()
#rects1 = ax.bar(x, ll, width, bottom=offset)
#
## Add some text for labels, title and custom x-axis tick labels, etc.

#%%

import seaborn as sns
sns.set(style="whitegrid", font_scale=3)
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
tips = sns.load_dataset("tips")
ax = plt.figure(figsize=(10,8))
ax = sns.barplot(x=x, y=ll, bottom=offset, palette=sns.color_palette())

ax.set_ylabel('Log-Likelihood')
ax.set_title('Linkedin', fontsize=50)
#ax.set_title('Mathoverflow', fontsize=50)
ax.set_xticks(x)
ax.set_xticklabels(('MLE-Sep', 'MLE-All','MTL', 'HARMLESS\n   (FOMAML)', 'HARMLESS\n     (MAML)', 'HARMLESS\n     (Reptile)'),rotation=45)
#ax.legend()
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)
plt.savefig('../result/linkedin_ll.pdf')
#plt.savefig('../result/math_ll.pdf')
plt.show()


#%%
delta_T_list = list(np.arange(1,11,1))
delta_T_list = list(np.arange(0.25,1,0.25))+delta_T_list
delta_T_list = [item/47.7753 for item in delta_T_list]

baseline1_auc = [0.79911634756995575,
                 0.61201386858264761,
                 0.5967173327664399,
                 0.5935830413052221,
                 0.58608381114301333,
                 0.61035552787006275,
                 0.64334760273972602,
                 0.66236645299145303,
                 0.68577453987730064,
                 0.68835739423974718,
                 0.74429285424827774,
                 0.76249999999999996,
                 0.72050905531081744]

baseline_mtl = [0.65183823529411766,
                 0.49709678766865784,
                 0.5149164476787278,
                 0.54979605905670126,
                 0.58120388349514562,
                 0.62390417648072627,
                 0.66813505424616537,
                 0.68873904853033519,
                 0.74914451171360885,
                 0.74098244995149487,
                 0.75624746724300951,
                 0.69358407079646023,
                 0.59338235294117647]

maml_auc = [0.5946980854197349, 0.560127097930585, 0.5912912045921573, 
            0.6235342401500938, 0.6307552390886443, 0.6115898909066037, 
            0.6650538295368729, 0.6712512171372931, 0.6891287180837062, 
            0.7217440119760479, 0.7356378600823045, 0.7463235294117647, 0.5857771260997067]
maml_auc5 = [0.6094256259204712, 0.558293954700994, 0.5724572457245725, 0.6009498123827393, 0.6009198037981777, 0.5969415090683173, 0.6449445898241979, 0.6791626095423564, 0.7128718083706239, 0.6970434131736527, 0.7710288065843621, 0.8036764705882353, 0.685483870967742]

reptile_auc = [0.6527245949926362, 0.5577643799902232, 0.5369461474449331, 
               0.5538109756097561, 0.5647828073411894, 0.5770151499606717, 
               0.6186181703226259, 0.67289435248296, 0.6770729139247171, 
               0.6337949101796407, 0.6747325102880658, 0.7625, 0.686950146627566]


plt.plot(delta_T_list, baseline1_auc, label = 'baseline 1')
plt.plot(delta_T_list, baseline_mtl, label = 'baseline mtl')
plt.plot(delta_T_list, maml_auc, label = 'maml K=3')
plt.plot(delta_T_list, maml_auc5, label = 'maml K=5')
plt.plot(delta_T_list, reptile_auc, label = 'reptile K=5')

plt.legend()
plt.show()








