'''
多层社区检测统计分析
网络稀疏化5%
w = 1,γ = 1
'''
# 1. Q值使用SPSS进行双样本t检验，原始数据均服从正态分布，无显著性差异结果。

import os
import numpy as np
from scipy import stats
from statsmodels.stats import multitest
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import statsmodels
import scipy


# 2. 社区划分数量统计分析
def CommunityNumberAnalysis(root_path,DM_number,HC_number):
	community_numbers = []
	for file in os.listdir(root_path):
		community_assignment = np.loadtxt(root_path + "/" + file)
		community_number = np.max(community_assignment)
		community_numbers.append(community_number)

	# 社区数量的双样本t检验，查看统计分析结果

	# 正态性检验
	norm_flag = False
	norm_p_value_DM = stats.shapiro(community_numbers[0:DM_number])[1]
	norm_p_value_HC = stats.shapiro(community_numbers[DM_number:])[1]

	if norm_p_value_DM > 0.05 and norm_p_value_HC > 0.05:
		norm_flag = True
		print("两组数据均符合正态分布")

	# 方差齐性检验
	variance_flag = False
	variance_p_value = stats.levene(community_numbers[0:DM_number],community_numbers[DM_number:])[1]

	if variance_p_value > 0.05:
		variance_flag = True
		print("两组数据符合方差齐性")

	if norm_flag and variance_flag:
		print("两组数据符合双样本t检验的前提")

		# 双样本t检验
		p_value = stats.ttest_ind(community_numbers[0:DM_number],community_numbers[DM_number:])[1]

		if p_value < 0.05:
			print("两组数据的社区数量存在差异")
		else:
			print("两组数据的社区数量不存在差异")

	else:
		print("数据不符合双样本t检验前提，需重新非参数检验")
		# Mann-Whitney U检验
		p_value = scipy.stats.mannwhitneyu(community_numbers[0:DM_number],community_numbers[DM_number:])[1]

		if p_value < 0.05:
			print("--------------")
			print("两组数据的社区数量存在差异")
		else:
			print("--------------")
			print("两组数据的社区数量不存在差异")

# 3.灵活性指标统计分析
def flexibilityAnalysis(root_path,DM_number,HC_number):
	flexibility_all = [] # 120*246维矩阵
	for file in os.listdir(root_path):
		community_assignment = np.loadtxt(root_path +"/" + file)
		flexibility = []
		for i in range(0,len(community_assignment)):
			count = 0
			for j in range(0,len(community_assignment[i])):
				if j < len(community_assignment[i]) - 1:
					if community_assignment[i][j] != community_assignment[i][j + 1]:
						count = count + 1
			flex = count / (len(community_assignment[0]))
			flexibility.append(flex)
		flexibility_all.append(flexibility)
	return flexibility_all

# 全脑水平灵活性指标统计分析
def GlobalFlexibilityAnalysis(flexibility_all,DM_number,HC_number):
	# 全脑水平的flexibility统计分析
	flex_values_all = [] # 1*120维向量
	for t in range(0,len(flexibility_all)):
		flex_all = np.mean(flexibility_all[t])
		flex_values_all.append(flex_all)

	# 正态性检验
	norm_flag = False
	norm_p_value_DM = stats.shapiro(flex_values_all[0:DM_number])[1]
	norm_p_value_HC = stats.shapiro(flex_values_all[DM_number:])[1]

	if norm_p_value_DM > 0.05 and norm_p_value_HC > 0.05:
		norm_flag = True
		print("两组数据均符合正态分布")

	# 方差齐性检验
	variance_flag = False
	variance_p_value = stats.levene(flex_values_all[0:DM_number],flex_values_all[DM_number:])[1]

	if variance_p_value > 0.05:
		variance_flag = True
		print("两组数据符合方差齐性")

	if norm_flag and variance_flag:
		print("两组数据符合双样本t检验的前提")

		# 双样本t检验
		p_value = stats.ttest_ind(flex_values_all[0:DM_number],flex_values_all[DM_number:])[1]

		if p_value < 0.05:
			print("两组数据的全脑水平灵活性存在差异")
		else:
			print("两组数据的全脑水平灵活性不存在差异")
	else:
		print("数据不符合双样本t检验前提，需重新非参数检验")
		# Mann-Whitney U检验
		p_value = scipy.stats.mannwhitneyu(flex_values_all[0:DM_number],flex_values_all[DM_number:])[1]

		if p_value < 0.05:
			print("--------------")
			print("两组数据的全脑水平灵活性存在差异")
		else:
			print("--------------")
			print("两组数据的全脑水平灵活性不存在差异")

# 节点水平灵活性指标统计分析
def NodalFlexibilityAnalysis(flexibility_all,DM_number,HC_number):
	# 单脑区节点水平的flexibility统计分析
	p_values = []
	for t in range(len(flexibility_all[0])):
		flex_values_all = np.array(flexibility_all)[:,t]

		# 正态性检验
		norm_flag = False
		norm_p_value_DM = stats.shapiro(flex_values_all[0:DM_number])[1]
		norm_p_value_HC = stats.shapiro(flex_values_all[DM_number:])[1]

		if norm_p_value_DM > 0.05 and norm_p_value_HC > 0.05:
			norm_flag = True
			#print("两组数据均符合正态分布")

		# 方差齐性检验
		variance_flag = False
		variance_p_value = stats.levene(flex_values_all[0:DM_number],flex_values_all[DM_number:])[1]

		if variance_p_value > 0.05:
			variance_flag = True
			#print("两组数据符合方差齐性")

		if norm_flag and variance_flag:
			print("两组数据符合双样本t检验的前提")

			# 双样本t检验
			p_value = stats.ttest_ind(flex_values_all[0:DM_number],flex_values_all[DM_number:])[1]
			p_values.append(p_value)

		else:
			print("数据不符合双样本t检验前提，需重新进行非参数检验")
			# Mann-Whitney U检验
			p_value = scipy.stats.mannwhitneyu(flex_values_all[0:DM_number],flex_values_all[DM_number:])[1]
			p_values.append(p_value)

	# bh多重比较校正
	p_values_corrected = statsmodels.stats.multitest.multipletests(p_values,alpha = 0.05,method = 'fdr_bh')[1]

	flag = False
	for p_value_corrected in p_values_corrected:
		if p_value_corrected < 0.05:
			flag = True

	if flag:
		print("--------------")
		print("两组数据的节点水平灵活性存在差异")

	else:
		print("--------------")
		print("两组数据的节点水平灵活性不存在差异")

# 3. 模块忠诚度指标分析
def moduleAllegianceMatrixAnalysis(root_path,node_number,DM_number,HC_number):
	module_allegiance_avg = {}
	subject_id = 0
	for file in os.listdir(root_path):
		community_assignment = np.loadtxt(root_path + "/" +file)
		module_allegiance_matrix_all = {}
		count = 0
		subject_id = subject_id + 1
		for w in range(0,len(community_assignment[0])):
			count = count + 1
			module_allegiance_matrix = []
			for i in range(0,len(community_assignment)):
				module_allegiance_row = []
				for j in range(0,len(community_assignment)):
					if community_assignment[i][w] == community_assignment[j][w]:
						module_allegiance_value = 1
					else:
						module_allegiance_value = 0
					module_allegiance_row.append(module_allegiance_value)
				module_allegiance_matrix.append(module_allegiance_row)
			module_allegiance_matrix_all[str(count)] = module_allegiance_matrix

		# 所有滑动窗下的平均模块忠诚度指标
		module_allegiance_matrix_avg = []
		for i in range(0,node_number):
			module_allegiance_row_avg = []
			for j in range(0,node_number):
				values = []
				for value in module_allegiance_matrix_all.values():
					values.append(value[i][j])
				module_allegiance_row_avg.append(np.mean(values))
			module_allegiance_matrix_avg.append(module_allegiance_row_avg)

		module_allegiance_avg[str(subject_id)] = module_allegiance_matrix_avg

	# 滑动窗平均水平上的模块忠诚度统计分析
	p_values = []
	for i in range(0,node_number):
		for j in range(0,node_number):
			values = []
			for value in module_allegiance_avg.values():
				values.append(value[i][j])

			# 正态性检验
			norm_flag = False
			norm_p_value_DM = stats.shapiro(values[0:DM_number])[1]
			norm_p_value_HC = stats.shapiro(values[DM_number:])[1]

			if norm_p_value_DM > 0.05 and norm_p_value_HC > 0.05:
				norm_flag = True
				#print("两组数据均符合正态分布")

			# 方差齐性检验
			variance_flag = False
			variance_p_value = stats.levene(values[0:DM_number],values[DM_number:])[1]

			if variance_p_value > 0.05:
				variance_flag = True
				#print("两组数据符合方差齐性")

			if norm_flag and variance_flag:
				print("两组数据符合双样本t检验的前提")

				# 双样本t检验
				p_value = stats.ttest_ind(values[0:DM_number],values[DM_number:])[1]
				p_values.append(p_value)

			else:
				print("数据不符合双样本t检验前提，需重新进行非参数检验")
				# Mann-Whitney U检验
				p_value = scipy.stats.mannwhitneyu(values[0:DM_number],values[DM_number:])[1]
				p_values.append(p_value)

	# bh多重比较校正
	p_values_corrected = statsmodels.stats.multitest.multipletests(p_values,alpha = 0.05,method = 'fdr_bh')[1]

	flag = False
	for p_value_corrected in p_values_corrected:
		if p_value_corrected < 0.05:
			flag = True

	if flag:
		print("--------------")
		print("两组数据的模块忠诚度连接存在差异")

	else:
		print("--------------")
		print("两组数据的模块忠诚度连接不存在差异")



if __name__ == '__main__':
	root_path = "I:/T2DM/CommunityDetectionAnalysis/246_50TR_1TR_120_0.05_sparsity_results"
	DM_number = 68
	HC_number = 52
	node_number = 246
	CommunityNumberAnalysis(root_path,DM_number,HC_number) # 社区数量检测
	flexibility_all = flexibilityAnalysis(root_path,DM_number,HC_number) # 灵活性指标统计分析
	GlobalFlexibilityAnalysis(flexibility_all,DM_number,HC_number) # 全脑水平的灵活性指标分析
	NodalFlexibilityAnalysis(flexibility_all,DM_number,HC_number) # 节点水平的灵活性指标分析
	moduleAllegianceMatrixAnalysis(root_path,node_number,DM_number,HC_number) # 模块忠诚度指标分析
