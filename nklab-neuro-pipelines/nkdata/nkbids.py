"""
=====
Parcellation Pipeline
=====
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import nibabel as nb
import nibabel.cifti2 as cif
import nibabel.gifti as gif
import numpy as np
import os
import sys
import glob
import shutil
from distutils.dir_util import copy_tree
from multiprocessing import cpu_count

__version__=0.1


def get_rootpath(filedir):
	return filedir.split('/sub')[0]

def sessions_exist(filedir)
	session=get_attribute(filedir.split('/'),"ses")
	return session

def get_attribute(bidsname, bidsattr):
	try:
		value=next(x for x in bidsname if x.startswith(bidsattr))
	except StopIteration:
		value=""
	return value

def get_bidstype_from_directory(filedir):
	if sessions_exist(filedir):
		bidstype=filedir.split('/sub')[2]
	else:
		bidstype=filedir.split('/sub')[1]
	return bidstype

def get_bids(filename):
	biddict={}
	biddict['filename']=filename
	biddict['filedir']=""

	#if "filename" is an absolute path then we need to grab the filename and directory
	if os.path.isabs(filename):
		filedir=os.path.dirname(os.path.abspath(filename))
		filename=os.path.basename(filename)
			biddict['filename']=filename
			biddict['filedir']=filedir
	bidsname=filename.split('_')
	subject=get_attribute(bidsname,"sub")
	session=get_attribute(bidsname,"ses")
	run=get_attribute(bidsname,"run")
	task=get_attribute(bidsname,"task")
	acq=get_attribute(bidsname,"acq")
	modality=bidsattrs[-1].split('.')[0]
	biddict['subject']=subject
	biddict['session']=session
	biddict['run']=run
	biddict['task']=modality
	biddict['acq']=modality
	biddict['modality']=modality

	#if the full path was passed then we can try and provide directory path
	if biddict['filedir']:
		datatype=get_bidstype_from_directory(filedir)
		biddict['datatype']=datatype
		biddict['projectdir']=get_rootpath(filedir)
		biddict['subdir']=os.path.join(biddict['projectdir'],biddict['subject'])
		if sessions_exist(filedir):
			biddict['sessiondir']=os.path.join(biddict['subdir'],biddict['session'])
			biddict['datadir']=os.path.join(biddict['sessiondir'],biddict['datatype'])
		else:
			biddict['sessiondir']=""
			biddict['datadir']=os.path.join(biddict['subdir'],biddict['datatype'])

	return biddict


def get_parser():
	from argparse import ArgumentParser
	from argparse import RawTextHelpFormatter

	parser = ArgumentParser(description="BIDS conversion using BIDSkit. You will need to clone"
		"BIDSkit from its repository and will also need dcm2niix installed.",formatter_class=RawTextHelpFormatter)
	parser.add_argument('bids_dir', action='store',
		help='The directory where BIDSkit will place the bids files')
	parser.add_argument('--version', action='version', version='parcellation pipeline v{}'.format(__version__))

	# Options for workflow
	g_work = parser.add_argument_group('suppress workflow options')
	g_work.add_argument('--participant_label', action='store', type=str, nargs='*', default=1, help="Subjects to process")
	g_work.add_argument('-s','--steps', action='store', type=int, nargs='*', default=[1,2,3,4,5,6,7,8], help="What processing to perform.")
	return parser


def main():
	# Run parser
	opts = get_parser().parse_args()

	#get a list of all subjects and cycle through each one
	

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
	main()
