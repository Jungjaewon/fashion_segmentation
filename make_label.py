import json
import h5py
import random
import subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import matplotlib as plt


def get_all_ctgs(h5pyfile, mode):
	#print(h5pyfile.keys()) # <KeysViewHDF5 ['#refs#', 'all_category_name', 'all_colors_name', 'fashion_dataset']>

	if mode == 'color':
		refs = h5pyfile.get('all_colors_name').value[0]
	else:
		refs = h5pyfile.get('all_category_name').value[0]
	all_ctgs = []
	for ref in refs:
		#print(h5pyfile[ref].value)
		ctg = ''.join([chr(c[0]) for c in h5pyfile[ref].value])
		all_ctgs.append(ctg)
	return all_ctgs


def make_ctg_tsv(mat_file='./fashon_parsing_data.mat', mode='color'):
	f = h5py.File(mat_file, 'r')
	all_ctgs = get_all_ctgs(f, mode)

	# save
	sr = pd.Series(all_ctgs)
	sr.name = 'category'
	sr.index.name = 'category_id'
	if mode == 'color':
		sr.to_csv('./label/colors_category.tsv', sep='\t', header=True)
	else:
		sr.to_csv('./label/cloth_category.tsv', sep='\t', header=True)


def make_seg_json(mat_file='./fashon_parsing_data.mat', mode='color'):
	assert mode in ['color', 'category']
	f = h5py.File(mat_file, 'r')
	all_ctgs = get_all_ctgs(f, 'color')
	iter_ = iter(f.get('#refs#').values())
	df = pd.DataFrame()

	for outfit in tqdm(iter_, total=len(f.get('#refs#'))):
		try:
			#print(outfit.keys()) # <KeysViewHDF5 ['category_label', 'color_label', 'img_name', 'segmentation']>

			# super pix 2 category
			if mode == 'color':
				spix2ctg = outfit.get('color_label').value[0]
			else:
				spix2ctg = outfit.get('category_label').value[0]
			#pd.Series(spix2ctg).value_counts().plot(kind='bar')

			# img_name
			ascii_codes = list(outfit.get('img_name').value[:,0])
			img_name = ''.join([chr(code) for code in ascii_codes])
			#print(f'img_name : {img_name}')

			# super pix
			spixseg = outfit.get('segmentation').value.T
			#plt.imshow(spixseg)
			#print(f'spixseg : {np.shape(spixseg)}') # spixseg : (600, 400)

			# super pix -> semantic segmentation
			semseg = np.zeros(spixseg.shape)
			#print(f'spix2ctg : {spix2ctg}')
			for i, c in enumerate(spix2ctg):
				semseg[spixseg == i] = c-1

			df = df.append({
				'img_name': img_name,
				'semseg': semseg,
			}, ignore_index=True)
		except AttributeError:
			pass

	d = df.to_dict(orient='records')
	#print(type(d)) # list
	random.seed(1234)
	random.shuffle(d)
	num_data = len(d)
	cut_t = int(len(d) * 0.9)
	train_d = d[:cut_t]
	test_d = d[cut_t:]

	if mode == 'color':
		with open('./label/color_segment_train.plk', 'wb') as f:
			pickle.dump(train_d, f)
		with open('./label/color_segment_test.plk', 'wb') as f:
			pickle.dump(test_d, f)
	else:
		with open('./label/category_segment_train.plk', 'wb') as f:
			pickle.dump(train_d, f)
		with open('./label/category_segment_test.plk', 'wb') as f:
			pickle.dump(test_d, f)


def make_bbox_json(mat_file='./fashon_parsing_data.mat'):
	f = h5py.File(mat_file, 'r')
	all_ctgs = get_all_ctgs(f)
	iter_ = iter(f.get('#refs#').values())
	df = pd.DataFrame()
	for outfit in tqdm(iter_, total=len(f.get('#refs#'))):
		try:
			# super pix 2 category
			spix2ctg = outfit.get('category_label').value[0]
			#pd.Series(spix2ctg).value_counts().plot(kind='bar')

			# img_name
			ascii_codes = list(outfit.get('img_name').value[:,0])
			img_name = ''.join([chr(code) for code in ascii_codes ])

			# super pix
			spixseg = outfit.get('segmentation').value.T
			#plt.imshow(spixseg)

			# super pix -> semantic segmentation
			semseg = np.zeros(spixseg.shape)
			#print(f'spix2ctg : {spix2ctg}')
			for i, c in enumerate(spix2ctg):
				semseg[spixseg == i] = c-1

			# semseg -> bbox
			items = []
			for i, ctg in enumerate(all_ctgs):
				region = np.argwhere(semseg == i)
				if region.size != 0:
					bbox = {
						'ymin':int(region.min(0)[0]),
						'xmin':int(region.min(0)[1]),
						'ymax':int(region.max(0)[0]),
						'xmax':int(region.max(0)[1]),
					}
					items.append({
						'bbox': bbox,
						'category': ctg,
					})

			df = df.append({
				'img_name': img_name,
				'items': items,
			}, ignore_index=True)
		except AttributeError:
			pass

	d = df.to_dict(orient='records')
	with open('./label/bbox.json', 'w') as f:
		json.dump(d, f, indent=4)


if __name__=='__main__':
	subprocess.call(['mkdir', 'label'])
	make_ctg_tsv(mode='color')
	make_ctg_tsv(mode='category')
	make_seg_json(mode='color')
	make_seg_json(mode='category')
	#make_bbox_json()
