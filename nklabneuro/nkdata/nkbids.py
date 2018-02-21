from __future__ import print_function
import os

__version__=0.1


def get_rootpath(filedir):
	return filedir.split('/sub')[0]

def sessions_exist(filedir):
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
		bidstype=filedir.split('/sub')[2].split('/')[1]
	else:
		bidstype=filedir.split('/sub')[1].split('/')[1]
	return bidstype

def get_bids(filename):
	bids_dict={}
	bids_dict['filename']=filename
	bids_dict['filedir']=""

	#if "filename" is an absolute path then we need to grab the filename and directory
	if os.path.isabs(filename):
		filedir=os.path.dirname(os.path.abspath(filename))
		filename=os.path.basename(filename)
		bids_dict['filename']=filename
		bids_dict['filedir']=filedir
	bidsname=filename.split('_')
	subject=get_attribute(bidsname,"sub")
	session=get_attribute(bidsname,"ses")
	run=get_attribute(bidsname,"run")
	task=get_attribute(bidsname,"task")
	acq=get_attribute(bidsname,"acq")
	modality=bidsname[-1].split('.')[0]
	bids_dict['subject']=subject
	bids_dict['session']=session
	bids_dict['run']=run
	bids_dict['task']=task
	bids_dict['acq']=acq
	bids_dict['modality']=modality

	#if the full path was passed then we can try and provide directory path
	if bids_dict['filedir']:
		datatype=get_bidstype_from_directory(filedir)
		bids_dict['datatype']=datatype
		bids_dict['projectdir']=get_rootpath(filedir)
		bids_dict['subdir']=os.path.join(bids_dict['projectdir'],bids_dict['subject'])
		if sessions_exist(filedir):
			bids_dict['sessiondir']=os.path.join(bids_dict['subdir'],bids_dict['session'])
			bids_dict['datadir']=os.path.join(bids_dict['sessiondir'],bids_dict['datatype'])
		else:
			bids_dict['sessiondir']=""
			bids_dict['datadir']=os.path.join(bids_dict['subdir'],bids_dict['datatype'])

	return bids_dict


def main():
	print("This module is meant to be imported. It provides a collection of calls for managing BIDS structure")

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
	main()
