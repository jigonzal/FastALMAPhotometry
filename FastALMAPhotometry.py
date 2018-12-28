import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from sklearn.cluster import DBSCAN
from astropy.modeling import models, fitting
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import os
import argparse


def GetBeam(file):
	head = fits.open(file)[0].header
	#Jy/beam to Jy/pix
	bmaj = head['BMAJ']*3600.0
	bmin = head['BMIN']*3600.0
	bpa = head['BPA']
	pix_size = abs(head['CDELT2']*3600.0)
	factor = 2*(np.pi*bmaj*bmin/(8.0*np.log(2)))/(pix_size**2)
	factor = 1.0/factor
	return bmaj,bmin,factor,bpa,pix_size


def GetDetections(fo,sigma):
	f = fo.flatten()
	f = f[~np.isnan(f)]
	detections = np.where(fo>=abs(args.MinSN)*sigma)
	return sigma,abs(args.MinSN),detections,fo	


def GetSourcesForSigma(detections,fo,sn_lim,plot_name,sigma):
	X = []
	for i in range(len(detections[0])):
		X.append(np.array([detections[0][i],detections[1][i]]))
	X = np.array(X)
	db = DBSCAN(eps=10, min_samples=1).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	fo2 = fo*1.0
	fo2[fo<sigma*sn_lim]=0
	fo2[0][0]=-1*max(fo.flatten())
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	centro_x = []
	centro_y = []
	flux_point_source = []

	for k, col in zip(unique_labels, colors):
		if k == -1:
			col = 'k'
		class_member_mask = (labels == k)
		xy = X[class_member_mask & core_samples_mask]
		if k>=0:
			aux = []
			for i in range(len(xy)):
				aux.append(fo[ xy[i][0]][xy[i][1]])
			flux_point_source.append(np.max(aux))
			centro_x.append(np.median(xy[:, 1]))
			centro_y.append(np.median(xy[:, 0]))
		xy = X[class_member_mask & ~core_samples_mask]

	flux_point_source = np.array(flux_point_source)
	centro_x = np.array(centro_x)
	centro_y = np.array(centro_y)
	return flux_point_source[np.argsort(flux_point_source)[::-1]],centro_x[np.argsort(flux_point_source)[::-1]],centro_y[np.argsort(flux_point_source)[::-1]]
	
def GetFitsForSigma(x_c,y_c,factor,fo,sigma,plot_name,bmaj,bmin,bpa,fijo,pix_size):
	ancho = int(round(0.5*(5.0*bmaj/pix_size))*2)
	center = int(ancho/2.0)
	z = fo[max(int(y_c)-center,0):min(int(y_c)+center,len(fo)),max(int(x_c)-center,0):min(int(x_c)+center,len(fo[0]))]
	y,x = np.mgrid[0:len(z),0:len(z[0])]
	p_init = models.Gaussian2D(amplitude=np.nanmax(z.flatten()),x_mean=center,y_mean=center,x_stddev=(bmaj/2.355)/pix_size,y_stddev=(bmin/2.355)/pix_size,theta=(bpa*2.0*np.pi/360.0)+np.pi/2) 
	
	if fijo:
		p_init.x_stddev.fixed = True
		p_init.y_stddev.fixed = True
		p_init.theta.fixed = True
	fit_p = fitting.LevMarLSQFitter()
	with warnings.catch_warnings():
	    warnings.simplefilter('ignore')
    	p = fit_p(p_init, x, y, z)
	
	model_flat = p(x, y).flatten()
	model2 = model_flat[model_flat>=0.135*max(model_flat)]
	peak_model = np.max(model_flat)
	fo[max(int(y_c)-center,0):min(int(y_c)+center,len(fo)),max(int(x_c)-center,0):min(int(x_c)+center,len(fo[0]))] = fo[max(int(y_c)-center,0):min(int(y_c)+center,len(fo)),max(int(x_c)-center,0):min(int(x_c)+center,len(fo[0]))] - np.nan_to_num(p(x, y))
	# plt.imshow(fo[max(int(y_c)-center,0):min(int(y_c)+center,len(fo)),max(int(x_c)-center,0):min(int(x_c)+center,len(fo[0]))],origin='lower')
	# plt.contour(p(x, y),origin='lower')
	# plt.show()
	return np.sum(model_flat)*factor,np.sqrt(len(model2)*factor)*sigma,peak_model,np.std(z - p(x, y)),p.x_stddev.value,p.y_stddev.value,p.theta.value,fo


def GetBestSigma(data):
	fo = 1.0*data
	f = fo.flatten()
	f = f[~np.isnan(f)]
	sigma = np.std(sigma_clip(f,sigma=5.0,iters=None))
	detections = np.where(fo>=abs(args.SecureSN)*sigma)

	if len(detections[0])==0:
		return sigma
	else:
		flux_point_source,centro_x,centro_y = GetSourcesForSigma(detections,fo,args.SecureSN,'lala',sigma)
		bmaj,bmin,factor,bpa,pix_size = GetBeam(args.Image)
		for i in range(len(centro_x)):
			aux_fits = GetFitsForSigma(centro_x[i],centro_y[i],factor,data,sigma,'lala',bmaj,bmin,bpa,False,pix_size)
			data = aux_fits[-1]
		sigma2 = np.std(data[~np.isnan(data)])
		aux = fits.open(args.Image)
		aux[0].data[0][0] = data
		aux.writeto('Residual.fits',overwrite=True)
		return sigma2
	
def GetSources(detections,fo,sn_lim,plot_name,sigma):
	plt.figure()
	X = []
	for i in range(len(detections[0])):
		X.append(np.array([detections[0][i],detections[1][i]]))
	X = np.array(X)
	db = DBSCAN(eps=2, min_samples=1).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	fo2 = fo*1.0
	fo2[fo<sigma*sn_lim]=0
	fo2[0][0]=-1*max(fo.flatten())
	plt.imshow(fo2,cmap='coolwarm',origin='lower')
	unique_labels = set(labels)
	colors = plt.cm.get_cmap('flag')(np.linspace(0, 1, len(unique_labels)))
	
	centro_x = []
	centro_y = []
	flux_point_source = []
	for k, col in zip(unique_labels, colors):
		if k == -1:
			col = 'k'
		class_member_mask = (labels == k)
		xy = X[class_member_mask & core_samples_mask]
		if k>=0:
			aux = []
			for i in range(len(xy)):
				aux.append(fo[xy[i][0]][xy[i][1]])
			flux_point_source.append(np.max(aux))
			centro_x.append(np.median(xy[:, 1]))
			centro_y.append(np.median(xy[:, 0]))
		plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,markeredgecolor='none', markersize=3)
		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 1], xy[:, 0], 'p', markerfacecolor=col,markeredgecolor='none', markersize=2)

	plt.savefig('plots/'+plot_name+'_sources.pdf')
	plt.clf()
	plt.close()
	flux_point_source = np.array(flux_point_source)
	centro_x = np.array(centro_x)
	centro_y = np.array(centro_y)
	return flux_point_source[np.argsort(flux_point_source)[::-1]],centro_x[np.argsort(flux_point_source)[::-1]],centro_y[np.argsort(flux_point_source)[::-1]]
	
def GetFits(x_c,y_c,factor,fo,sigma,plot_name,bmaj,bmin,bpa,fijo,pix_size):
	plt.figure()
	ancho = int(round(0.5*(5.0*bmaj/pix_size))*2)
	center = int(ancho/2.0)
	z = fo[max(int(y_c)-center,0):min(int(y_c)+center,len(fo)),max(int(x_c)-center,0):min(int(x_c)+center,len(fo[0]))]
	y,x = np.mgrid[0:len(z),0:len(z[0])]
	p_init = models.Gaussian2D(amplitude=max(z.flatten()),x_mean=center,y_mean=center,x_stddev=(bmaj/2.355)/pix_size,y_stddev=(bmin/2.355)/pix_size,theta=(bpa*2.0*np.pi/360.0)+np.pi/2) 
	if fijo:
		p_init.x_stddev.fixed = True
		p_init.y_stddev.fixed = True
		p_init.theta.fixed = True

	fit_p = fitting.LevMarLSQFitter()
	with warnings.catch_warnings():
	    warnings.simplefilter('ignore')
    	p = fit_p(p_init, x, y, z)

	model_flat = p(x, y).flatten()
	model2 = model_flat[model_flat>=0.135*max(model_flat)]
	peak_model = np.max(model_flat)
	chisquare = z - p(x, y)
	chisquare = chisquare.flatten()
	chisquare = chisquare[z.flatten()>=2.0*sigma]
	ll = 1.0*len(chisquare)
	chisquare = chisquare/(np.ones(int(ll))*sigma)
	chisquare = chisquare*chisquare

	chisquare = np.sum(chisquare)
	if fijo:
		chisquare = chisquare/(ll-3.0)
	else:
		chisquare = chisquare/(ll-6.0)	
	w, h = 1.0*plt.figaspect(0.33)
	fig = plt.figure(figsize=(w,h))
	plt.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00,wspace=0.00, hspace=0.0)
	ax1 = plt.subplot(1,3,1)
	plt.imshow(z, origin='lower', interpolation='nearest',vmin=-max(z.flatten()), vmax=max(z.flatten()),cmap='coolwarm')
	plt.contour(z,colors='black',levels=np.append(np.arange(args.MinSN,50,np.sqrt(2))[::-1]*-1,np.arange(args.MinSN,50,np.sqrt(2)))*sigma,linewidths=1)
	plt.title("Data")
	plt.text(0.5, 0.9,' ', horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes,bbox=dict(facecolor='yellow', alpha=1),fontsize=25)
	plt.setp( plt.gca().get_xticklabels(), visible=False)
	plt.setp( plt.gca().get_yticklabels(), visible=False)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	ax2 = plt.subplot(1,3,2,sharey=ax1)
	plt.imshow(p(x, y), origin='lower', interpolation='nearest',vmin=-max(z.flatten()), vmax=max(z.flatten()),cmap='coolwarm')
	plt.contour(p(x, y),colors='black',levels=np.append(np.arange(2,50,np.sqrt(2))[::-1]*-1,np.arange(2,50,np.sqrt(2)))*sigma,linewidths=1)

	plt.title("Model")
	if fijo:
		plt.text(0.5, 0.9,'3 Parameters', horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes,bbox=dict(facecolor='yellow', alpha=1),fontsize=25)
	else:
		plt.text(0.5, 0.9,'6 Parameters', horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes,bbox=dict(facecolor='yellow', alpha=1),fontsize=25)
	plt.setp( plt.gca().get_xticklabels(), visible=False)
	plt.setp( plt.gca().get_yticklabels(), visible=False)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	ax3 = plt.subplot(1,3,3,sharey=ax1)
	plt.imshow(z - p(x, y), origin='lower', interpolation='nearest',vmin=-max(z.flatten()), vmax=max(z.flatten()),cmap='coolwarm')
	plt.contour(z - p(x, y),colors='black',levels=np.append(np.arange(2,50,np.sqrt(2))[::-1]*-1,np.arange(2,50,np.sqrt(2)))*sigma,linewidths=1)

	plt.title("Residual")
	plt.text(0.5, 0.9,r'$\chi^2_{\rm red}=%.1f$'%(chisquare), horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes,bbox=dict(facecolor='yellow', alpha=1),fontsize=25)
	plt.setp( plt.gca().get_xticklabels(), visible=False)
	plt.setp( plt.gca().get_yticklabels(), visible=False)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	plt.savefig('plots/'+plot_name)
	plt.clf()
	plt.close()
	return np.sum(model_flat)*factor,np.sqrt(len(model2)*factor)*sigma,peak_model,chisquare,p.x_stddev.value,p.y_stddev.value,p.theta.value*180.0/np.pi

	
def RunSearch(output):
	hdulist =   fits.open(args.Image,memmap=True)
	data  = 1.0*hdulist[0].data[0][0]  
	pb = fits.open(args.PBImage,memmap=True)[0].data[0][0]
	data = np.where(pb<args.PBLimit,np.nan,data)
	data = np.where(np.isnan(pb),np.nan,data)

	bmaj,bmin,factor,bpa,pix_size = GetBeam(args.Image)
	w = WCS(args.Image)
	data_aux = 1.0*data
	sigma = GetBestSigma(data)
	print 'RMS:',round(sigma*1e6,1),' microJy/beam'
	data = 1.0*data_aux
	data_aux = 0
	sigma,sn_lim,detections,fo = GetDetections(data,sigma)

	fo  = 1.0*hdulist[0].data[0][0] 
	fo = np.nan_to_num(fo)

	if len(detections[0])==0:
		print 'no detections',detections
		return output
	flux_point_source,centro_x,centro_y = GetSources(detections,fo,args.MinSN,'Candidate',sigma)	
	flux_response = pb
	for i in range(len(centro_x)):
		print 'Measuring properties for candidate:',i+1,'/',len(centro_x)
		model_name = 'Candidate_ID'+str(i+1)+'.pdf'
		aux = 'ID' + str(i+1).zfill(2) +'\t'
		dec = w.all_pix2world(centro_x[i], centro_y[i],0,0, 0)[w.wcs.lat]
		ra= w.all_pix2world(centro_x[i], centro_y[i],0,0, 0)[w.wcs.lng]
		c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs').to_string('hmsdms',sep=':',precision=3)
		ra,dec = c.split()
		aux = aux + ra + '\t' + dec + '\t' + str(round(sn_lim,1)) + '\t'		
		aux = aux + '%.1f\t%.1f\t%.1f\t'%(flux_point_source[i]*1e6,sigma*1e6,flux_point_source[i]/sigma)
			
		aux_fits = GetFits(centro_x[i],centro_y[i],factor,fo,sigma,model_name.replace('.pdf','_3p.pdf'),bmaj,bmin,bpa,True,pix_size)
		aux_fits2 = GetFits(centro_x[i],centro_y[i],factor,fo,sigma,model_name.replace('.pdf','_6p.pdf'),bmaj,bmin,bpa,False,pix_size)
		
		aux = aux + '%.1f\t%.1f\t%.1f\t'%(aux_fits[2]*1e6,aux_fits[0]*1e6,aux_fits[1]*1e6)
		aux = aux + '%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f\n'%(aux_fits2[2]*1e6,aux_fits2[0]*1e6,aux_fits2[1]*1e6,aux_fits[3],aux_fits2[3],aux_fits[4],aux_fits[5],aux_fits[6],aux_fits2[4],aux_fits2[5],aux_fits2[6],flux_response[int(centro_y[i])][int(centro_x[i])])
		output.write(aux)
		output.flush()

	return output


def main():

	#Parse the input arguments
	parser = argparse.ArgumentParser(description="Python script that makes a fast photometry of an ALMA continuum image")
	parser.add_argument('-Image', type=str, required=True,help = 'Continuum Image')
	parser.add_argument('-PBImage', type=str, required=True , help = 'Associated PB image, or flux image')
	parser.add_argument('-MinSN', type=float, default = 3.0, required=False,help = 'Minimum S/N value to save in the outputs. ')
	parser.add_argument('-SecureSN', type=float, default = 5.0, required=False,help = 'Minimum S/N value to assume a source is secure. Will remove all sources with S/N higher than this value to get real rms')
	parser.add_argument('-PBLimit', type=float, default = 0.5,required=False,help = 'PB limit to do the search')
	global args

	args = parser.parse_args()
	#Checking input arguments
	print 20*'#','Checking inputs....',20*'#'
	if os.path.exists(args.Image):
	    print '*** Cube',args.Image,'found ***'
	else:
	    print '*** Cube',args.Image,'not found ***\naborting..'
	    exit()

	if os.path.exists(args.PBImage):
	    print '*** Cube',args.PBImage,'found ***'
	else:
	    print '*** Cube',args.PBImage,'not found ***\naborting..'
	    exit()

	output = open('TableFastPhot.dat','w')
	if os.path.isdir('plots/'):
		os.system('rm plots/*')
	else:
		os.mkdir('plots')

	header = ''
	header = header +'#1:ID..................ID given to each source in each field, not sorted\n'
	header = header +'#2:RA..................Right ascencion of the central pixel\n'
	header = header +'#3:DEC.................Declination of the central pixel\n'
	header = header +'#4:SN_limit............signal-to-noise of the minimum value in the image\n'
	header = header +'#5:S...................Peak pixel in each detections[microJy]\n'
	header = header +'#6:S_err...............Sigma estimated for each image[microJy]\n'
	header = header +'#7:SN..................signal-to-noise of the detection (Estimated with S and S_err)\n'
	header = header +'#8:S_fit_peak_3p.......Peak of the model fitted to the detection, 3 parameteres[microJy]\n'
	header = header +'#9:S_fit_int_3p.......Integrated flux over the model of 3 parameteres[microJy]\n'
	header = header +'#10:S_fit_int_err_3p...Error estimation in the previous quantity[microJy]\n'
	header = header +'#11:S_fit_peak_6p......Peak of the model fitted to the detection, 6 parameteres[microJy]\n'
	header = header +'#12:S_fit_int_6p.......Integrated flux over the model of 6 parameteres[microJy]\n'
	header = header +'#13:S_fit_int_err_6p...Error estimation in the previous quantity[microJy]\n'
	header = header +'#14:reduced_chi2_3p....Reduced chi squared for 3 parameters best fit\n'
	header = header +'#15:reduced_chi2_6p....Reduced chi squared for 6 parameters best fit\n'
	header = header +'#16:sigma_x_3p.........Sigma of the 2D gaussian in x-axis for the 3 parameters model\n'
	header = header +'#17:sigma_y_3p.........Sigma of the 2D gaussian in y-axis for the 3 parameters model\n'
	header = header +'#18:theta_3p...........angle of the 2D gaussian for the 3 parameters model\n'
	header = header +'#19:sigma_x_6p.........Sigma of the 2D gaussian in x-axis for the 6 parameters model\n'
	header = header +'#20:sigma_y_6p.........Sigma of the 2D gaussian in y-axis for the 6 parameters model\n'
	header = header +'#21:theta_6p...........angle of the 2D gaussian for the 6 parameters model\n'
	header = header +'#22:PB.................Primary Beam correction, divide by this number to get the real flux density\n'
	output.write(header)
	output = RunSearch(output)
	output.close()

main()

