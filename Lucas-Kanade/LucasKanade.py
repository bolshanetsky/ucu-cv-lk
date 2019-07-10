import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(temp, frame, roi, p0 = np.zeros(2)):
	# Input: 
	#	temp: template image
	#	frame: Current image
	#	roi: Current position of the car
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    THRESHOLD = 0.1
    x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3]
    Iy, Ix = np.gradient(frame)
    delta_p = 1
    while np.square(delta_p).sum() > THRESHOLD:
        
        #STEP 1 warp image
        px, py = p0[0], p0[1]
        x1_w, y1_w, x2_w, y2_w = x1+px, y1+py, x2+px, y2+py
        
        x = np.arange(0, temp.shape[0], 1)
        y = np.arange(0, temp.shape[1], 1)
        
        c = np.linspace(x1, x2, 87)
        r = np.linspace(y1, y2, 36)
        cc, rr = np.meshgrid(c, r)
    
        cw = np.linspace(x1_w, x2_w, 87)
        rw = np.linspace(y1_w, y2_w, 36)
        ccw, rrw = np.meshgrid(cw, rw)
        
        spline = RectBivariateSpline(x, y, temp)
        T = spline.ev(rr, cc)
        
        spline1 = RectBivariateSpline(x, y, frame)
        warpImg = spline1.ev(rrw, ccw)
        
        #STEP2 compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        
        #STEP3 compute gradient
        spline_gx = RectBivariateSpline(x, y, Ix)
        Ix_w = spline_gx.ev(rrw, ccw)

        spline_gy = RectBivariateSpline(x, y, Iy)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        #evaluate jacobian (2,2)
        jac = np.array([[1,0],[0,1]])
        
        #computer Hessian
        delta = I @ jac 
        #H is (2,2)
        H = delta.T @ delta
        
        #compute delta_p
        #dp is (2,2)@(2,n)@(n,1) = (2,1)
        delta_p = np.linalg.inv(H) @ (delta.T) @ errImg
        
        #update parameters
        p0[0] += delta_p[0,0]
        p0[1] += delta_p[1,0]
        
    p = p0
    return p
