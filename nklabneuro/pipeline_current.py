# 
# 
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys
from os import listdir,  makedirs
from os.path import isfile, isdir, join, exists, abspath
import numpy as np
import subprocess 
#import glob   ----- for wildcard recognition of filenames
#import shutil ----- for copying files
import nibabel as nb
from nilearn import plotting as niplot
import matplotlib.pyplot as plt

from nipype import Node, Function, Workflow, IdentityInterface, SelectFiles
#from nipype.interfaces import fsl
from nipype.interfaces.fsl import BET, ApplyMask, IsotropicSmooth
from nipype.interfaces.io import DataSink

from nipype.workflows.fmri.fsl import create_susan_smooth
import nbformat as nbf

from bids.grabbids import BIDSLayout



#open a python notebook for logging 
def openIPythonNB(fname):
#check to see if file exists
	if isfile(fname):
		try:
			with open(fname, 'r') as f:
				nbr = nbf.read(f,nbf.NO_CONVERT)
				return nbr
		except Exception as e:
			print ('Problem opening existing file:{0} due to error {1})'.format(fname,e))
			print ('\nLogging to python notebook cannot be done.')
			return None
	else:
		cells = []
		nbr = nbf.v4.new_notebook(cells=cells)
		return nbr

def addCell(nbr, text,code):
	if len(text)>0:
		nbr["cells"].append(nbf.v4.new_markdown_cell(text))
	if len(code)>0:
		nbr["cells"].append(nbf.v4.new_code_cell(code))

def getIPythonNB(fname, overwrite):
	if overwrite:
		cells=[]
		nbr = nbf.v4.new_notebook(cells=cells)
	else:
		nbr = openIPythonNB(fname)
	return nbr

def closeIPythonNB(fname, nbr, generate):
	try:
		with open(fname, 'w') as f:
			nbf.write(nbr, f)
		if generate:
			print("converting to jupyter notebook")
			subprocess.call(["jupyter nbconvert --execute --inplace " + fname], shell=True)
	except Exception as e:
		print ('Problem writing to file:{0} due to error {1})'.format(fname,e))

#open html for writing - this is a very basic implementation
# look at plotly, yattag etc for more dynamic and interative approaches
# ideally using d3.js to manipulate data.
def openHTML(fname):
	print("not yet implemented.")



#used by SUsan Workflow as it outputs a list
# see https://miykael.github.io/nipype_tutorial/notebooks/basic_workflow.html 
extract_func = lambda list_out: list_out[0]
list_extract = Node(Function(input_names=["list_out"],
                             output_names=["out_file"],
                             function=extract_func),
                    name="list_extract")

def getNodeOutputs(nodeLocation, valid):
	nodeOutputs = {}
	nodeLocation = join(nodeLocation,'_report')
	report = abspath(join(nodeLocation,'report.rst'))
	if exists(report):
		with open(report, 'r') as f:
			filetext = f.readlines()
			startInd = (filetext.index('Execution Outputs\n'))+3
			endInd   = (filetext.index('Runtime info\n'))-1
			for n in range(startInd,endInd):
				if len(filetext[n].strip())>0:
					param = filetext[n].split(':')[0][1:].strip()
					value = filetext[n].split(':')[1].strip()
					if not (valid and value == '<undefined>'):
						nodeOutputs[param]=value
	return nodeOutputs

def getNodeInputs(nodeLocation, valid):
	nodeInputs = {}
	nodeLocation = join(nodeLocation,'_report')
	report = abspath(join(nodeLocation,'report.rst'))
	if exists(report):
		with open(report, 'r') as f:
			filetext = f.readlines()
			startInd = (filetext.index('Execution Inputs\n'))+3
			endInd   = (filetext.index('Execution Outputs\n'))-1
			for n in range(startInd,endInd):
				if len(filetext[n].strip())>0:
					param = filetext[n].split(':')[0][1:].strip()
					value = filetext[n].split(':')[1].strip()
					if not (valid and value == '<undefined>'):
						nodeInputs[param]=value
	return nodeInputs


def getWorkFlowOutputs(workFlow, valid):
	wfOutputs = {}
	nodes = workFlow.list_node_names()
	for node in nodes:
		#to support embedded workflows - this is hokey - doesn't work for more nested workflows!!!
		if '.' in node:
			subNode=node.split('.')[-1]
			node = node.replace('.','/')
			node = node + '/mapflow/_'+subNode+'0'
		nodeLoc = join(workFlow.base_dir,workFlow.name)
		nodeLoc = join(nodeLoc,node)
		ndOutputs = getNodeOutputs(nodeLoc,valid)
		if len(ndOutputs) > 0:
			wfOutputs[node]=ndOutputs
	return wfOutputs



def getWorkFlowInputs(workFlow, valid):
	wfInputs = {}
	nodes = workFlow.list_node_names()
	for node in nodes:
		#to support embedded workflows - this is hokey - doesn't work for more nested workflows!!!
		if '.' in node:
			subNode=node.split('.')[-1]
			node = node.replace('.','/')
			node = node + '/mapflow/_'+subNode+'0'
		nodeLoc = join(workFlow.base_dir,workFlow.name)
		nodeLoc = join(nodeLoc,node)
		ndInputs = getNodeInputs(nodeLoc,valid)
		if len(ndInputs) > 0:
			wfInputs[node]=ndInputs
	return wfInputs



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
#extend function below to be more flexible and show ranges etc based on actual slice numbers 
# or fractions, or words (end, first, last) etc....
def plot_slice(fname, slice):
	#load the image
	img = nb.load(fname)
	data = img.get_data()
	maxSlices = data.shape[-1]

	if is_number(slice):
		sliceNum = min(slice,maxSlices)
	elif slice == 'mid':
		sliceNum = int(maxSlices/2)

	plt.imshow(np.rot90(data[...,sliceNum]),cmap="gray")
	plt.gca().set_axis_off()

def plot_slices(fnames,ranges, ori):
	subjects = len(fnames)
	slices = len(ranges)
	ORIENTATION='v'
	if ori == 'h':
		ORIENTATION='h'
		#figzise is width by height
		f = plt.figure(figsize=(slices,subjects))
	else:
		f = plt.figure(figsize=(subjects, slices))
	for i,img in enumerate(fnames):
		for j,slice in enumerate(ranges):
			if ORIENTATION == 'h':
				#f.add_subplot(1,1,1)
				subplot = f.add_subplot(subjects,slices,j+1+i*slices)
				subplot.text(0,0,'slice '+str(slice),backgroundcolor='yellow')
				#if j==0:
				#	f.text(0,(subjects - i)/(subjects)- 0.05,img.split("/")[-1],backgroundcolor='green')
				plot_slice(img,slice)
			else:
				subplot = f.add_subplot(slices,subjects,i+1+j*subjects)
				subplot.text(0,0,'slice '+str(slice),backgroundcolor='yellow')
				#if j==0:
				#	f.text(i/subjects,0.95,img.split("/")[-1],backgroundcolor='green')
				plot_slice(img,slice)
	plt.show(block=False)


#A very specific Shell Function
def juliaSmoothFunction(Command, in_file, out_file):
	shellCommandList = ['julia',Command, in_file, out_file]
	import subprocess # we have to import subprocess here for this to work
	#p = subprocess.Popen(shellCommandList)
	#out, err = p.communicate()
	out = subprocess.check_output(shellCommandList)
	return out


def shellCallFunction(shellCommand):
	subprocess.call([shellCommand], shell=True)

#for testing Nodes
def shellFunction(shellCommandList):
	import subprocess # we have to import subprocess here for this to work
	#p = subprocess.Popen(shellCommandList)
	#out, err = p.communicate()
	out = subprocess.check_output(shellCommandList)
	return out

def simpleFunction(inputParam):
	return inputParam + 2

#for use in BIDS workflow
def get_niftis(subject_id, data_dir, type):
    # Remember that all the necesary imports need to be INSIDE the function for the Function Interface to work!
    from bids.grabbids import BIDSLayout  
    layout = BIDSLayout(data_dir)  
    imgs = [f.filename for f in layout.get(subject=subject_id, type=type, extensions=['nii', 'nii.gz'])]   
    return imgs

def getArray(r,x,y):
	import nkutils  # we have to import here for this to work
	return nkutils.getArray(r,x,y)

def getInput(query):
	try:
		print(query)
		answer=input()
	except IOError as (errno, strerror):
		print ('I/O Error ({0}),{1}',format(errno,strerror))
		raise SystemExit
	except ValueError as (errno, strerror):
		print ('Value Error ({0}),{1}',format(errno,strerror))
		raise SystemExit
	except:
		print (sys.exc_info()[0])
		print ('unexplained error - breaking out of program')
		raise SystemExit
	else:
		pass
	finally:
		pass
	return answer

#function to demonstrate different ways of setting up and using nodes
def testNodes():
	#Node that calls a simple function
	print("\n ** debug PIPELINE : simple function Node.\n")
	simpleNode = Node(Function(input_names=["inputParam"], output_names=["simpleResult"], function=simpleFunction), name='simpleNode')
	simpleNode.inputs.inputParam = 5
	tempResult = simpleNode.run()
	#print(tempResult.outputs) -- this is equvalent to below
	print(simpleNode.result.outputs)

	#Node that calls a shell function
	TEST_DIR = abspath('/home/chidi/')
	print("\n ** debug PIPELINE : shell function Node.\n")
	shellPopNode = Node(Function(input_names=["shellCommandList"], output_names=["shellResult"], function=shellFunction), name='shellPopNode')
	shellPopNode.inputs.shellCommandList = ['ls', TEST_DIR, '-a']
	tempResult = shellPopNode.run()
	#print(tempResult.outputs) -- this is equivalent to below
	print(shellPopNode.result.outputs)

	#Node that calls a python module that prints to std 
	print("\n ** debug PIPELINE : python external module Node.\n")
	PYTHON='python'
	SCRIPT=abspath('/home/chidi/uabiomed/develop/nipype/nkutils.py')
	pythonExtNode = Node(Function(input_names=["shellCommandList"], output_names=["shellResult"], function=shellFunction), name='pythonExtNode')
	pythonExtNode.inputs.shellCommandList = [PYTHON, SCRIPT]
	tempResult = pythonExtNode.run()
	#print(tempResult.outputs) -- this is equivalent to below
	print(pythonExtNode.result.outputs)

	#Node that calls an imported python module
	print("\n ** debug PIPELINE : python imported module Node.\n")
	pythonIntNode = Node(Function(input_names=["r","x","y"], output_names=["shellResult"], function=getArray), name='pythonIntNode')
	pythonIntNode.inputs.r = 15
	pythonIntNode.inputs.x = 3
	pythonIntNode.inputs.y = 5
	tempResult = pythonIntNode.run()
	#print(tempResult.outputs) -- this is equivalent to below
	print(pythonIntNode.inputs)
	print(pythonIntNode.outputs)
	print(pythonIntNode.result.outputs)

def createNiPypeNode(function, nodeName, inputParams):
	NNode = Node(function, name=nodeName)
	for inputParam in inputParams:
		exec('NNode.inputs.'+ str(inputParam)+ '=inputParams[str(inputParam)]')
	return NNode

def anatPipeline(resultsDir, workDir, subDir, subid):

	pnf = join(subDir,subid+".ipynb")
	pnb = getIPythonNB(pnf, True)

	text= "# Anatomic Pipeline results for subject " + subid
	code= ""
	addCell(pnb, text,code)

	print("\n ** PIPELINE : starting anatomic pipeline.\n")
	ANAT_DIR = abspath(join(subDir, 'anat'))
	ANAT_T1W = abspath(join(ANAT_DIR,  subid + '_T1w.nii.gz'))
	ANAT_BET_T1W=abspath(join(resultsDir, subid + '_T1w_bet.nii.gz'))

	text="Anatomic T1W image"
	code=("%pylab inline\nimport nibabel as nb;img = nb.load('"
		+ANAT_T1W+"');data = img.get_data();"
	"plt.imshow(np.rot90(data[...,100]),cmap='gray');"
	"plt.gca().set_axis_off()")
	addCell(pnb, text,code)

	#A. PREPROCESSING PIPELINE

	#1. SKULL STRIPPING WITH BET
	betNodeInputs={}
	betNodeInputs['in_file']=ANAT_T1W
	#betNodeInputs['out_file']=ANAT_BET_T1W #very strange that this stops it working in workflow mode
	betNodeInputs['mask']=True
	betNodeInputs['frac']=0.5
	betNode = createNiPypeNode(BET(),'betNode',betNodeInputs)
	#n_file=ANAT_T1W
	#betNode = Node(BET(in_file=in_file,mask=True), name="betNode")
	#print(betNode.inputs)
	#print(betNode._interface.cmdline)
	#print(betNode.outputs)
	#betResults = betNode.run()


	#print('** debug ** command line output')
	#print(betNode._interface.cmdline)
	#print('** debug ** inputs')
	#print(betNode.inputs)
	#print('** debug ** outputs')
	#print(tempresult.outputs)

	#2. SMOOTH ORIGINAL IMAGE WITH ISOTROPIC SMOOTH
	smoothNodeInputs = {}
	smoothNodeInputs['in_file']=ANAT_T1W
	smoothNodeInputs['fwhm']=4
	smoothNode = createNiPypeNode(IsotropicSmooth(),'smoothNode',smoothNodeInputs)
	#smoothResults =smoothNode.run()

	#3. MASK SMOOTHED IMAGE WITH APPLYMASK
	maskNodeInputs = {}
	maskNode = createNiPypeNode(ApplyMask(),'maskNode',maskNodeInputs)

	# ILLUSTRATING USING WORKFLOW WITH APPLYMASK
	#4. create workflow
	wfName='smoothflow'
	wfGraph='smoothWorkflow_graph.dot'
	wfGraphDetailed='smoothWorkflow_graph_detailed.dot'
	wf = Workflow(name=wfName,base_dir=workDir)
	WF_DIR=abspath(join(workDir, wfName))
	wf.connect(betNode, "mask_file",maskNode, "mask_file")
	wf.connect([(smoothNode,maskNode,[("out_file", "in_file")])])
	wf.write_graph(wfGraph, graph2use='colored')
	wfImg = plt.imread(WF_DIR + '/' + wfGraph+'.png')
	plt.imshow(wfImg)
	#plt.show(block=False) #set to true if you want to see this graph
	wf.write_graph(wfGraph, graph2use='flat')
	wfImgDetailed = plt.imread(WF_DIR + '/' + wfGraphDetailed+'.png')
	#plt.imshow(wfImgDetailed)
	#plt.show(block=False)

	# run the workflow
	wf.run()
	print(wf.inputs)
	print(wf.outputs)
	#print(betNode.inputs)
	#print(betNode.outputs)
	#print(wf.get_node('betNode').inputs)
	#print(wf.get_node('betNode').outputs)

	pnames=[]
	wfInputs = getWorkFlowInputs(wf,True)
	if len(wfInputs)> 0:
		for node in wf.list_node_names():
			for key in wfInputs[node]:
				filename = wfInputs[node][key]
				filecomps = filename.split('.')
				if filecomps[-1]=='nii' or (filecomps[-1]=='gz' and filecomps[-2]=='nii'):
					pnames.append(filename)

	wfOutputs = getWorkFlowOutputs(wf,True)
	if len(wfOutputs)> 0:
		for node in wf.list_node_names():
			for key in wfOutputs[node]:
				filename = wfOutputs[node][key]
				filecomps = filename.split('.')
				if filecomps[-1]=='nii' or (filecomps[-1]=='gz' and filecomps[-2]=='nii'):
					pnames.append(filename)

	pnames = set(pnames)

	#
	#%pylab inline
	#from nilearn import plotting

	#niplot.plot_anat(betNode.inputs.in_file,title='T1W in-file', cut_coords=(10,10,10), display_mode='ortho', dim=-1, draw_cross=False, annotate=False)
	#niplot.show() #need a better way to display
	#plt.show(block=False)

	#niplot.plot_anat(smoothResults.outputs.out_file,title='T1W in-file', cut_coords=(10,10,10), display_mode='ortho', dim=-1, draw_cross=False, annotate=False)
	#niplot.show() #need a better way to display

	#niplot.plot_anat(betResults.outputs.out_file,title='T1W skull-stripped out-file', cut_coords=(10,10,10), display_mode='ortho', dim=-1, draw_cross=False, annotate=False)
	#niplot.show() #need a better way to display that doesn't hold up process
	#niplot.show() #for now just do this at the end.
	#plt.show(block=False)

	#plot images vertically
	#fnames=[]
	#fnames.append(betResults.outputs.out_file)
	#fnames.append(smoothResults.outputs.out_file)

	ranges = [30, 138, 180]
	plot_slices(pnames,ranges, 'v')
	plt.show()

	#plot images horizontally
	#fnames=[]
	#fnames.append(betResults.outputs.out_file)
	#fnames.append(smoothResults.outputs.out_file)

	ranges=range(130,136)
	plot_slices(pnames,ranges, 'h')
	plt.show()

	# ILLUSTRATING USING NESTED WORKFLOW WITH PROVIDED SUSAN FLOW for Non-linear smoothing
	# 5. Create Susan workflow and display
	# Note that to specify specific location for susan to work then you will need to embed it into another workflow!!!	
	wfName='susan_smooth'
	wfGraph='susan_smooth_graph.dot'
	WF_DIR=abspath(join(workDir, wfName))
	susanWf = create_susan_smooth(name='susan_smooth', separate_masks=False)
	#print(susanWf.inputs)
	#print(susanWf.outputs)
	#print(susanWf.inputs.inputnode) # this specifies the visible inputs/outputs to external
	#print(susanWf.outputs.outputnode)
	graphLoc=join(WF_DIR,wfGraph)
	susanWf.write_graph(graphLoc,graph2use='colored')
	susanWfImg = plt.imread(join(WF_DIR,wfGraph+'.png'))
	plt.imshow(susanWfImg)
	plt.gca().set_axis_off()
	plt.show()

	# 6. Create new Workflow and use Susan as the smoothing step
	# Initiate workflow with name and base directory
	wfName='smoothSusanFlow'
	wfGraph='smoothSusanFlow_graph.dot'
	WF_DIR=abspath(join(workDir, wfName))
	wf2 = Workflow(name=wfName, base_dir=workDir)

	# Create new skullstrip and mask nodes
	betNodeInputs={}
	betNodeInputs['in_file']=ANAT_T1W
	betNodeInputs['mask']=True
	betNodeInputs['frac']=0.5
	betNode2 = createNiPypeNode(BET(),'betNode2',betNodeInputs)

	maskNodeInputs = {}
	maskNode2 = createNiPypeNode(ApplyMask(),'maskNode2',maskNodeInputs)


	# Connect the nodes to each other and to the susan workflow
	wf2.connect([(betNode2, maskNode2, [("mask_file", "mask_file")]),
             (betNode2, susanWf, [("mask_file", "inputnode.mask_file")]),
             (susanWf, list_extract, [("outputnode.smoothed_files",
                                     "list_out")]),
             (list_extract, maskNode2, [("out_file", "in_file")])
             ])

	# Specify the remaining input variables for the susan workflow
	susanWf.inputs.inputnode.in_files = abspath(ANAT_T1W)
	susanWf.inputs.inputnode.fwhm = 4

	#detailed graph showing workflow details embedded in overall workflow
	graphLoc=join(WF_DIR,wfGraph)
	wf2.write_graph(graphLoc, graph2use='colored')
	graphImgLoc=join(WF_DIR,wfGraph+'.png')
	wf2Img = plt.imread(graphImgLoc)
	plt.imshow(wf2Img)
	plt.gca().set_axis_off()
	plt.show()


	text="Demonstrating Nilearn plotting"
	plotTitle = 'dynamic title'
	code=("%pylab inline\nfrom nilearn import plotting;plotting.plot_anat('"+ ANAT_T1W + 
		"',title='" + plotTitle +"', cut_coords=(" + "10,10,10"+"), display_mode='ortho', dim=-1, draw_cross=False, annotate=False)")
	addCell(pnb, text,code)

	#niplot.plot_anat(betNode.inputs.in_file,title='T1W in-file', cut_coords=(10,10,10), display_mode='ortho', dim=-1, draw_cross=False, annotate=False)
	#niplot.show() #need a better way to display
	#plt.show(block=False)

	text="Anatomic Workflow"
	code=("%pylab inline\nimg=matplotlib.image.imread('"+ graphImgLoc + 
		"');imgplot = plt.imshow(img)")
	addCell(pnb, text,code)
	closeIPythonNB(pnf, pnb, True)

	#graph showing summary of embedded workflow
	wf2.write_graph(join(WF_DIR,wfGraph), graph2use='orig')
	wf2Img = plt.imread(join(WF_DIR,wfGraph+'.png'))
	plt.imshow(wf2Img)
	plt.gca().set_axis_off()
	plt.show()

	#run the new workflow with embedded susan
	wf2.run()

	print(wf2.inputs)
	print(wf2.outputs)
	print(str(wf2.list_node_names()))
	print(str(susanWf.list_node_names()))

	#LOrdie there has to be an easier way than this to get the outputs and input files generically from a workflow?
	pnames=[]
	wfInputs = getWorkFlowInputs(wf2,True)
	if len(wfInputs)> 0:
		for node in wf2.list_node_names():
			if node in wfInputs:
				for key in wfInputs[node]:
					filename = wfInputs[node][key]
					filecomps = filename.split('.')
					if filecomps[-1]=='nii' or (filecomps[-1]=='gz' and filecomps[-2]=='nii'):
						pnames.append(filename)

	wfOutputs = getWorkFlowOutputs(wf2,True)
	if len(wfOutputs)> 0:
		for node in wf2.list_node_names():
			if node in wfOutputs:
				for key in wfOutputs[node]:
					filename = wfOutputs[node][key]
					filecomps = filename.split('.')
					if filecomps[-1]=='nii' or (filecomps[-1]=='gz' and filecomps[-2]=='nii'):
						pnames.append(filename)

	pnames = set(pnames)
	ranges = [30, 138, 180]
	plot_slices(pnames,ranges, 'v')
	plt.show()

	#plot images horizontally
	#fnames=[]
	#fnames.append(betResults.outputs.out_file)
	#fnames.append(smoothResults.outputs.out_file)

	ranges=range(130,136)
	plot_slices(pnames,ranges, 'h')
	plt.show()


	# 6 Demonstrate efficient recomputing of workflows - only dependent steps need to be recomputed
	#original workflow
	wf.inputs.smoothNode.fwhm = 1
	wf.run()

	pnames=[]
	wfOutputs = getWorkFlowOutputs(wf,True)
	if len(wfOutputs)> 0:
		for node in wf.list_node_names():
			for key in wfOutputs[node]:
				filename = wfOutputs[node][key]
				filecomps = filename.split('.')
				if filecomps[-1]=='nii' or (filecomps[-1]=='gz' and filecomps[-2]=='nii'):
					pnames.append(filename)
	pnames = set(pnames)
	ranges = [30, 138]
	plot_slices(pnames,ranges, 'h')
	plt.show()

	#susan workflow
	wf2.inputs.susan_smooth.inputnode.fwhm = 1
	wf2.run()

	pnames=[]
	wfOutputs = getWorkFlowOutputs(wf2,True)
	if len(wfOutputs)> 0:
		for node in wf2.list_node_names():
			if node in wfOutputs:
				for key in wfOutputs[node]:
					filename = wfOutputs[node][key]
					filecomps = filename.split('.')
					if filecomps[-1]=='nii' or (filecomps[-1]=='gz' and filecomps[-2]=='nii'):
						pnames.append(filename)

	pnames = set(pnames)
	ranges = [30, 138]
	plot_slices(pnames,ranges, 'h')
	plt.show()

def customPipeline(resultsDir, workDir, subDir, subid):
	DWI_DIR = abspath(join(subDir, 'dwi'))
	DWI_DATA = abspath(join(DWI_DIR,  subid + '_dwi.nii.gz'))
	DWI_SMOOTH= abspath(join(DWI_DIR,  subid + '_dwi_smooth.nii'))
	ANAT_DIR = abspath(join(subDir, 'anat'))
	ANAT_T1W = abspath(join(ANAT_DIR,  subid + '_T1w.nii.gz'))
	ANAT_SMOOTH=abspath(join(ANAT_DIR, subid + '_T1w_smooth.nii.gz'))
	#Node that calls a shell function
	RUN_DIR = abspath('/home/chidi/development/julia/denoiseJulia/testAlgo.jl')
	print("\n ** Custom Pipeline : Call Julia function from shell.\n")
	JuliaNode = Node(Function(input_names=["Command", "in_file", "out_file"], output_names=["shellResult"], function=juliaSmoothFunction), name='JuliaNode')
	JuliaNode.inputs.Command = RUN_DIR
	#JuliaNode.inputs.in_file =  DWI_DATA
	JuliaNode.inputs.out_file = ANAT_SMOOTH
	print(JuliaNode.outputs)
	print(JuliaNode.inputs)
	#tempResult = JuliaNode.run()
	#print(tempResult.outputs) -- this is equivalent to below - but only exists after a run!
	#print(JuliaNode.result.outputs)

	#creat workflow
	betNodeInputs={}
	betNodeInputs['mask']=True
	skullstrip = createNiPypeNode(BET(),'skullstrip',betNodeInputs)

	wfName = "NKSmooth"
	wf = Workflow(name=wfName)
	WF_DIR=abspath(join(workDir, wfName))
	wf.base_dir = WF_DIR

	# First, let's specify the list of input variables
	#subject_list = ['01', '02', '03', '04', '05']
	#session_list = ['retest', 'test']
	#fwhm_widths = [4, 8, 16]
	subject_list = ['10159', '10171', '10189', '10193']

	#infosource = Node(IdentityInterface(fields=['subject_id', 'session_id', 'fwhm_id']),
    #              name="infosource")
    #infosource.iterables = [('subject_id', subject_list),
    #                    ('session_id', session_list),
    #                    ('fwhm_id', fwhm_widths)]
	infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
	infosource.iterables = [('subject_id', subject_list)]

	# String template with {}-based strings
	templates = {'anat': 'sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz'}
	selectfiles = Node(SelectFiles(templates), name='selectfiles')
	# Location of the dataset folder
	selectfiles.inputs.base_directory = '/home/chidi/uabiomed/develop/nipype/ds30/'

	# Create DataSink object
	sinker = Node(DataSink(), name='sinker')
	# Name of the output folder
	sinker.inputs.base_directory = resultsDir

	wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                  (selectfiles, skullstrip,[('anat','in_file')]),
                  (skullstrip,JuliaNode,[('out_file','in_file')]),
                  (skullstrip,sinker,[('mask_file','test1.@mask_file')]),
                  (JuliaNode,sinker,[('out_file','test1.@out_file')])
                  ])

	print(wf.inputs)
	print(wf.outputs)
	#graph showing summary of embedded workflow
	wfGraph='NKsmoothflow_graph.dot'
	wf.write_graph(join(WF_DIR,wfGraph), graph2use='exec', format='png', simple_form=True)
	wfImg = plt.imread(join(WF_DIR,wfGraph+'.png'))
	plt.imshow(wfImg)
	plt.gca().set_axis_off()
	plt.show()

	# Define substitution strings
	#substitutions = [("_ses-test", "_ts"),
	#                 ('_ses-retest', '_rt'),
	#                ('_brain_mask', '_mask'),
	#                 ('_brain_smooth', '')]

	# Feed the substitution strings to the DataSink node
	#sinker.inputs.substitutions = substitutions

	# Run it in parallel (one core for each smoothing kernel)
	wf.run('MultiProc', plugin_args={'n_procs': 8})


def selectPipeline(resultsDir, workDir, subDir, subid):
	ANAT_DIR = abspath(join(subDir, 'ses-test/anat'))
	ANAT_T1W = abspath(join(ANAT_DIR,  subid + '_ses-test_T1w.nii.gz'))
	ANAT_BET_T1W=abspath(join(resultsDir, subid + '_ses-test_T1w_bet.nii.gz'))

	betNodeInputs={}
	betNodeInputs['mask']=True
	skullstrip = createNiPypeNode(BET(),'skullstrip',betNodeInputs)

	smoothNodeInputs = {}
	isosmooth = createNiPypeNode(IsotropicSmooth(),'isosmooth',smoothNodeInputs)


	# Create the workflow
	wfName = "iterSmooth"
	wf = Workflow(name=wfName)
	WF_DIR=abspath(join(workDir, wfName))
	wf.base_dir = WF_DIR

	# First, let's specify the list of input variables
	subject_list = ['01', '02', '03', '04', '05']
	session_list = ['retest', 'test']
	fwhm_widths = [4, 8, 16]

	infosource = Node(IdentityInterface(fields=['subject_id', 'session_id', 'fwhm_id']),
                  name="infosource")
	infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list),
                        ('fwhm_id', fwhm_widths)]

	# String template with {}-based strings
	templates = {'anat': 'sub-{subject_id}/ses-{ses_name}/anat/sub-{subject_id}_ses-{ses_name}_T1w.nii.gz'}
	#removed func as not being set up
	#	'func': 'sub-{subject_id}/ses-{ses_name}/func/sub-{subject_id}_ses-{ses_name}_task-{task_name}_bold.nii.gz'}
	# Create SelectFiles node
	selectfiles = Node(SelectFiles(templates), name='selectfiles')
	# Location of the dataset folder
	selectfiles.inputs.base_directory = '/home/chidi/uabiomed/develop/nipype/ds114/'
	# Feed {}-based placeholder strings with values
	#sf.inputs.subject_id = '01'
	#sf.inputs.ses_name = "test"
	#sf.inputs.task_name = 'fingerfootlips'
	#sf.run().outputsresultsDir

	# Create DataSink object
	sinker = Node(DataSink(), name='sinker')
	# Name of the output folder
	sinker.inputs.base_directory = resultsDir

	wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                             ('session_id', 'ses_name')]),
                  (infosource, isosmooth, [('fwhm_id', 'fwhm')]),
                  (selectfiles, skullstrip,[('anat','in_file')]),
                  (skullstrip,isosmooth,[('out_file','in_file')]),
                  (skullstrip,sinker,[('mask_file','test1.@mask_file')]),
                  (isosmooth,sinker,[('out_file','test1.@out_file')])
                  ])

	print(wf.inputs)
	print(wf.outputs)
	#graph showing summary of embedded workflow
	wfGraph='itersmoothflow_graph.dot'
	wf.write_graph(join(WF_DIR,wfGraph), graph2use='exec', format='png', simple_form=True)
	wfImg = plt.imread(join(WF_DIR,wfGraph+'.png'))
	plt.imshow(wfImg)
	plt.gca().set_axis_off()
	plt.show()

	# Define substitution strings
	substitutions = [("_ses-test", "_ts"),
	                 ('_ses-retest', '_rt'),
	                 ('_brain_mask', '_mask'),
	                 ('_brain_smooth', '')]

	# Feed the substitution strings to the DataSink node
	sinker.inputs.substitutions = substitutions

	# Run it in parallel (one core for each smoothing kernel)
	wf.run('MultiProc', plugin_args={'n_procs': 8})


def iterPipeline(resultsDir, workDir, subDir, subid):
	ANAT_DIR = abspath(join(subDir, 'ses-test/anat'))
	ANAT_T1W = abspath(join(ANAT_DIR,  subid + '_ses-test_T1w.nii.gz'))
	ANAT_BET_T1W=abspath(join(resultsDir, subid + '_ses-test_T1w_bet.nii.gz'))

	betNodeInputs={}
	betNodeInputs['in_file']=ANAT_T1W
	betNodeInputs['mask']=True
	skullstrip = createNiPypeNode(BET(),'skullstrip',betNodeInputs)

	smoothNodeInputs = {}
	isosmooth = createNiPypeNode(IsotropicSmooth(),'isosmooth',smoothNodeInputs)

	isosmooth.iterables = ("fwhm", [4, 8, 16])

	# Create the workflow
	wfName = "isosmoothflow"
	wf = Workflow(name=wfName)
	WF_DIR=abspath(join(workDir, wfName))
	wf.base_dir = WF_DIR
	wf.connect(skullstrip, 'out_file', isosmooth, 'in_file')

	# Run it in parallel (one core for each smoothing kernel)
	wf.run('MultiProc', plugin_args={'n_procs': 3})
	#graph showing summary of embedded workflow
	wfGraph='smoothflow_graph.dot'
	wf.write_graph(join(WF_DIR,wfGraph), graph2use='exec', format='png', simple_form=True)
	wfImg = plt.imread(join(WF_DIR,wfGraph+'.png'))
	plt.imshow(wfImg)
	plt.gca().set_axis_off()
	plt.show()

	# First, let's specify the list of input variables
	subject_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
	session_list = ['ses-retest', 'ses-test']
	fwhm_widths = [4, 8]




def bidsPipeline(resultsDir, workDir, subDir, subid):
	layout = BIDSLayout("/ds102/")
	subjects = layout.get_subjects()
	mods = layout.get_modalities()
	mod_types = layout.get_types(modality='func')
	tasks = layout.get_tasks()
	#layout.get(subject='01', modality="anat", session="test")



def restPipeline(resultsDir, workDir, subDir, subid):
	FUNC_DIR = join(subDir, 'func')
	anatPipeline(resultsDir, workDir, subDir, subid)
	print("\n ** PIPELINE : starting resting state MRI pipeline.\n")

def taskPipeline(resultsDir, workDir, subDir, subid):
	FUNC_DIR = join(subDir, 'func')
	anatPipeline(resultsDir, workDir, subDir, subid)
	print("\n ** PIPELINE : starting task fMRI pipeline.\n")

def dwiSetup():
	print("\n ** PIPELINE : the default dwi pipeline is presented below. \n")

def dwiPipeline(resultsDir, workDir, subDir, subid):
	DWI_DIR = join(subDir, 'dwi')
	anatPipeline(resultsDir, workDir, subDir, subid)
	print("\n ** PIPELINE : starting dMRI pipeline")

def tmsPipeline(resultsDir, workDir, subDir, subid):
	TMS_DIR = join(subDir, 'tms')
	anatPipeline(resultsDir, workDir, subDir, subid)
	print("\n ** PIPELINE : starting tms pipeline")

if __name__ == '__main__':
	#Study main Directory in BIDS format
	STUDY_DIR = abspath('/home/chidi/uabiomed/develop/nipype/ds102/')
	#Example STUDY_DIR below also has test, retest folders at patient level
	# and so SUB_DIR will have to be amended to ensure that data is retrievable
	#STUDY_DIR = abspath('/home/chidi/uabiomed/develop/nipype/ds114/')
	#view the Study Tree
	print("\n** PIPELINE : displaying directory structure for Study Directory = " + STUDY_DIR + '\n')
	shellCallFunction("tree -L 3 " + STUDY_DIR)
	#call(["tree -F 4 " + STUDY_DIR], shell=True)

	#identify subjects
	subs = [d for d in listdir(STUDY_DIR) if (isdir(join(STUDY_DIR, d)) and d.find('sub-')==0)]
	subs.sort()
	print("\n** PIPELINE : counted  " + str(len(subs))+ ' subjects\n')
	print(subs)

	PIPELINE = 'uabiomed_tmsfmri'
	SUB = subs[0]
	SUB_DIR = abspath(join(STUDY_DIR, SUB))
	RESULTS_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/out/'+SUB))
	if not exists(RESULTS_DIR):
		makedirs(RESULTS_DIR)

	WORK_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/work/'+SUB))
	if not exists(WORK_DIR):
		makedirs(WORK_DIR)

	pipeline = getInput('What pipeline would you like to run? [d=dwi,r=rsFmri,t=tFmri,a=anatomic; default=none]\nplease note that the anatomic pipeline will be run as first step of other pipelines.\n')
	if pipeline == 'r':
		restPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 't':
		taskPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'd':
		dwiPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'm':
		tmsPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'a':
		anatPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'b':
		bidsPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'c':
		STUDY_DIR = abspath('/home/chidi/uabiomed/develop/nipype/ds30/')
		SUB = 'sub-10159'
		SUB_DIR = abspath(join(STUDY_DIR, SUB))
		RESULTS_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/out/'+SUB))
		if not exists(RESULTS_DIR):
			makedirs(RESULTS_DIR)
		WORK_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/work/'+SUB))
		if not exists(WORK_DIR):
			makedirs(WORK_DIR)
		customPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'i':
		STUDY_DIR = abspath('/home/chidi/uabiomed/develop/nipype/ds114/')
		SUB = 'sub-01'
		SUB_DIR = abspath(join(STUDY_DIR, SUB))
		RESULTS_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/out/'+SUB))
		if not exists(RESULTS_DIR):
			makedirs(RESULTS_DIR)
		WORK_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/work/'+SUB))
		if not exists(WORK_DIR):
			makedirs(WORK_DIR)
		iterPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 's':
		STUDY_DIR = abspath('/home/chidi/uabiomed/develop/nipype/ds114/')
		SUB = 'sub-01'
		SUB_DIR = abspath(join(STUDY_DIR, SUB))
		RESULTS_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/out/'+SUB))
		if not exists(RESULTS_DIR):
			makedirs(RESULTS_DIR)
		WORK_DIR = abspath(join(STUDY_DIR,'results/'+PIPELINE+'/work/'+SUB))
		if not exists(WORK_DIR):
			makedirs(WORK_DIR)
		selectPipeline(RESULTS_DIR,WORK_DIR,SUB_DIR,SUB)
	elif pipeline == 'test':
		testNodes()
	else:
		print('no pipeline selected')
