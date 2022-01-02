# Generic sound class to store sound synthesizer files
import numpy as np
import math
from scipy.signal import butter, lfilter, sawtooth, windows
from opensimplex import OpenSimplex

from numpy.random import seed

dssynthseed=18005551212 #default
dssynthsr=44100 #default

class DSParam():
    '''
        Provides API for parameter creation, getting, setting
        @cb - a callback function to execute when the parameter changes.
        @synth_doc - documentation for mapping information from input units
    '''    
    def __init__(self,name,min,max, val, cb,synth_doc) :
        self.name=name
        self.min=min
        self.max=max
        self.val=val
        self.cb=cb
        self.synth_doc = synth_doc

    # only store the actual value, not the normed value used for setting
    def __setParamNorm__(self, i_val) :
        self.val=self.min + i_val * (self.max - self.min);

##################################################################################################
#  the base sound model from which all synths should derive
##################################################################################################
'''
    A model has parameters, methods to get/set them, and a generate function that returns a signal.
    This is *the* interface for all synths.
'''
class DSSoundModel() :
    '''
        @rngseed - If None, will use random seed
    '''
    def __init__(self,sr=dssynthsr, rngseed=dssynthseed, verbose=False) :
        self.param = {} # a dictionary of DSParams
        self.sr = sr 
        self.rng = np.random.default_rng(rngseed)
        self.verbose=verbose
        if verbose : 
            if rngseed==None :
                print(f"{self.__class__.__name__} created rng with random seed")
            elif rngseed==dssynthseed :
                print(f"{self.__class__.__name__} created rng with default seed")
            else :
                print(f"{self.__class__.__name__} created rng with user seed") 
                
            print(f"{self.__class__.__name__} initialized with sr={sr}")

    # -----------------------vv param interface vv-------------------------------------#
    # Create a parameter with an API for your DSSynth
    def __addParam__(self, name,min,max,val, cb=None, synth_doc="") :
        self.param[name]=DSParam(name,min,max,val, cb, synth_doc)

    # expose a parameter from another DSSynth as your own.
    def __addChildParam__(self, child, childParamName, val=None, newname=None, cb=None, synth_doc=None) :
        nombre=newname or childParamName #child.getParam(childParam, "name")
        self.__addParam__(nombre, 
            child.getParam(childParamName, "min"), 
            child.getParam(childParamName, "max"), 
            val or child.getParam(childParamName, "val"), 
            cb= cb or (lambda v :
                child.setParam(childParamName, v)), #self.getParam(nombre)), 
            synth_doc=synth_doc or child.getParam(childParamName,  "synth_doc"))

    ''' just set parameter to a value '''
    def setParam(self, name, value) :
        self.param[name].val=value
        if self.param[name].cb is not None :
            self.param[name].cb(value)
        assert self.param[name].val <= self.param[name].max and self.param[name].val >=  self.param[name].min, f"setParam({name}, {value})  param val {self.param[name]} ({self.param[name].val}) out of range [{self.param[name].min},{self.param[name].max}]"
        

    ''' set parameters using [0,1] which gets mapped to [min, max] '''
    def setParamNorm(self, name, nvalue) :
        self.param[name].__setParamNorm__(nvalue)
        if self.param[name].cb is not None :
            # Pass the "natural units" value of the parameter to the callback, not the normed.
            self.param[name].cb(self.getParam(name))
        assert self.param[name].val <= self.param[name].max and self.param[name].val >=  self.param[name].min, f"setParamNorm({name}, {nvalue}) : param val {self.param[name]} ({self.param[name].val}) out of range [{self.param[name].min},{self.param[name].max}]"
        

    def getParam(self, name, prop="val") :
        if prop == "val" :
            return self.param[name].val
        if prop == "min" :
            return self.param[name].min
        if prop == "max" :
            return self.param[name].max
        if prop == "name" :
            return self.param[name].name
        if prop == "synth_doc" :
            return self.param[name].synth_doc

    ''' returns list of paramter names that can be set by the user '''
    def getParams(self) :
        plist=[]
        for p in self.param :
            plist.append(self.param[p].name)
        return plist
    # -----------------------^^ param interface ^^-------------------------------------#

    '''
        override this for your signal generation
    '''
    def generate(self, sigLenSecs=1) :
        return np.zeros(sigLenSecs*self.sr)

    ''' returns list of paramter names and their ranges '''
    def paramProps(self) :
        plist=[]
        for p in self.param :
            plist.append(self.param[p])
        return plist

    ''' Print all the parameters and their ranges from the synth'''
    def printParams(self):
        paramVals = self.paramProps()
        for params in paramVals:
            print(f"Name: {params.name}, Current val: {params.val:.3f}, Min val: {params.min}, Max val: {params.max}, synth_doc: {params.synth_doc}")

##################################################################################################
# A Ensemble class for playing a bunch of DSSynth models together
##################################################################################################
'''
    A DSSynth for creating a bunch of models that play at the same time.
    Create each in the usual way, setting their parameters, etc. Then pass them as an array to DSEnsemble.
    The factory takes an optional argument for a list of amplitudes that, if used, must be the same length as the models list.
    The generate function takes a spreadSecs argument that lest you spread out start times evenly over an interval.
'''
class DSEnsemble(DSSoundModel) : 
    '''
        @rngseed - If None, will use random seed
    '''
    def __init__(self,  models=[], amp=[], sr=dssynthsr, rngseed=dssynthseed) :
        DSSoundModel.__init__(self, sr=sr, rngseed=dssynthseed)
        self.numModels= len(models)
        self.models=models
        if len(amp) != len(models) :
            print(f'will use uniform amplitudes unless len(amps) == len(models)')
            amp=np.ones(len(models))*.6
        self.amp=amp
        self.sr=sr

    #spreadSecs was a bad idea, but for backwards compatibility......
    def generate(self,  durationSecs, spreadSecs=0, verbose=False) :
        numSamples=int(self.sr*durationSecs)
        if verbose :
            print(f'Ensemble.generate with {self.sr=} and  {durationSecs=} will compute {numSamples=}')
        if spreadSecs==0 :
            spreadsamples=0
        else :
            spreadsamples=self.rng.integers(0,int(self.sr*spreadSecs))    
                  
        sig=np.zeros(numSamples+spreadsamples)
        for i in range (self.numModels) :
            gensig = self.amp[i]*self.models[i].generate(durationSecs) 
            sig = addin(gensig, sig, spreadsamples) 
        return sig[:numSamples]


##################################################################################################
# A couple of handy-dandy UTILITY FUNCTIONS for event pattern synthesizers in particular
##################################################################################################

def noisySpacingTimeList(rate_exp, irreg_exp, durationSecs,  rngseed, phase=None, verbose=False, wrap=True, roll=False) :
    '''
    creates a list of event times that happen with a rate of 2^r_exp and deviate from the strict equal space according to irreg_exp

    @phase - None randomizes, a number shifts the event list by [0-1]/eps seconds (prior to wrap and roll)
    @wrap - mode by duration so that anything that fell off either end is wrapped back in to [0,durationSecs]
    @roll - shift all events so that first one starts at time 0 (and phase parameter becomes irrelevant) 
    '''
    rng = np.random.default_rng(seed=rngseed)

    # mapping to the right range units
    eps=np.power(2.,rate_exp)
    irregularity=.1*irreg_exp*np.power(10,irreg_exp)
    sd=irregularity/eps

    linspacesteps=int(eps*durationSecs)
    linspacedur = linspacesteps/eps

    if verbose :
        print(f'noisySpacingTimeList: rate_exp is {rate_exp}, eps is {eps}, sd = {sd}, linspacesteps is {linspacesteps}, linspacedur is {linspacedur}')

    eventtimes=[(x+rng.normal(scale=sd))%durationSecs for x in np.linspace(0, linspacedur, linspacesteps, endpoint=False)]

    if verbose :
        print(f'noisySpacingTimeList: have {len(eventtimes)} over {durationSecs} for an average of {len(eventtimes)}/{durationSecs} eps')
        print(f'noisySpacingTimeList: (BEFORE wrapped, rolled) eventtimes =  {eventtimes}')

    if phase==None :
        phase=rng.random()
        if verbose :
            print(f'  noisySpacingTimeList:  phase is randomly set to {phase}')
    eventtimes = eventtimes + phase/eps

    if wrap :
        eventtimes=np.sort(np.mod(eventtimes, durationSecs))
    if roll :
        eventtimes=eventtimes-np.min(eventtimes)

    if verbose :
        print(f'noisySpacingTimeList: (wrapped, rolled) eventtimes =  {eventtimes}')


    return eventtimes #sort because we "wrap around" any events that go off the edge of [0. durationSecs]



''' convert a list of floats (time in seconds) to a signal with pulses at those time '''
def timeList2Sig(elist, sr, durationSecs) :
    numsamps=sr*durationSecs
    sig=np.zeros(numsamps)
    for nf in elist :
        sampnum=int(round(nf*sr))
        if sampnum<numsamps and sampnum >= 0 :
            sig[sampnum]=1
        else :
            print("in timeList2Sig, warning: sampnum(={}) out of range".format(sampnum))
    return sig



'''adds one (shorter) array (a) in to another (b) starting at startsamp in b'''
def addin(a,b,startsamp) :
    b[startsamp:startsamp+len(a)]=[sum(x) for x in zip(b[startsamp:startsamp+len(a)], a)]
    return b



def extendEventSequence(oseq, seqDur, durationSecs) :
    ''' 
    # Exend an event (time) list by concatenating the sequence with seDur added to the events in each successive repeat.
    # @ oseq - original list of times
    # @ seqDur - the duration of each revolution through the oseq
    # @ durationSecs - keep cycling through oseq to create the looped seq while less than durationSecs
    '''
    cyclelength=len(oseq)
    newEvList=[]
    newEvNum=0
    revNum=0
    revSeqEvNum=0
    t=oseq[revSeqEvNum]
    # Cycle through oseq events incrementing time while less than requested duration
    while t < durationSecs :
            newEvList.append(t)
            # now get the next one
            newEvNum=newEvNum+1
            revNum=newEvNum//cyclelength
            revSeqEvNum=newEvNum%cyclelength
            t=oseq[revSeqEvNum]+revNum*seqDur
    return newEvList



''' Create an array comprised of linear segments between breakpoints '''
# y - list of values
# s - list of number of samples to interpolate between sucessive values
def bkpoint(y,s) :
    assert(len(y)==(len(s)+1))
    sig=[]
    for j in range(len(y)-1) :
        sig=np.concatenate((sig, np.linspace(y[j], y[j+1], s[j], False)), 0)
    return sig


def env(sigLenSecs, sr, attack=0.005, decay=0.005) : 
    '''
        env(sigLenSecs, attack=0.005, decay=0.005)
        envelope with a linear attack and decay specified in seconds
    '''
    length = int(round(sigLenSecs*sr))   # in samples
    ltrans1 = round(min(attack*sr, length/2)) #in samples
    ltrans2 = min(length-ltrans1-1, round(min(decay*sr, length/2)))  # -1 for the zero point we add at the end of bkpoint
    mids=max(0, length-(ltrans1+ltrans2)-1)
    #print(f"calling bkpoint with ltrans1={ltrans1}, ltrans1={ltrans2},midms={midms}")
    return np.array(bkpoint([0,1,1,0,0],[ltrans1,mids,ltrans2,1]))


def oct2freq(octs, bf=440.) :
    return bf * np.power(2,octs)

def freq2oct(freq, bf=440.) :
    return np.log2(freq/bf)

def gwindow(m) :
    '''
        Gaussian window
        @m - the number of samples for your gaussian, uses sd=m/6 to get near-zero at tails 
        @return - array storing samples of a symmetric gaussian 
    '''
    return windows.gaussian(m,m/6)

def expWindow(dur, attack_s=0.005, decay_s=0.005, tscale=3, sr=dssynthsr) :
    tarray=np.linspace(0,dur, int(dur*sr), endpoint=True)
    return [expdecay(t, dur, attack_s, decay_s, tscale) for t in tarray]

def expdecay(t, dur, attack_s=0, decay_s=0, tscale=3) :
    '''
    Exponential decay window
    @t - time in seconds
    @dur - duration in seconds to go from 1 to exp(-tscale)
    @attack_s - linear ramp attack time in seconds
    @decay_s - linear ramp decay time in seconds
    @tscale - default=3, so signals decays from 1 to exp(-3) =.05 over the duration in seconds.
    '''
    scale=1
    if (attack_s!=0 and t<attack_s) : scale=t/attack_s
    if (decay_s!=0 and t>(dur-decay_s)) : scale=(dur-t)/decay_s
    return scale*np.exp(-tscale*t/(dur))
#########################################
# https://pypi.org/project/opensimplex/
def simplex(numsamples, freq, sr, octaveWeights=[1], ymin=-1, ymax=1, rngseed=dssynthseed, verbose=False) :
    ''' Uses OpenSimplex to create noise in the range ymin to ymax 
    @freq - number of bumps per second
    @sr - samplerate
    @octaveWeights - array of weghtings for summing simplex noise at oct*freq
    @ ymin, ymax - map simplex range [-1,1] to [ymin, ymax]
    '''
    s=int(rngseed/1000)
    if verbose : 
        print(f'simplex: {s=}')
    simplex=OpenSimplex(seed=s)
    freq=max(freq,0.0000001)
    sig=np.zeros((numsamples))
    for n in range(numsamples) :
        for oct in range(len(octaveWeights)) :
            feature_size=sr/(freq*2**oct)
            sig[n]=sig[n]+octaveWeights[oct]*simplex.noise2(n / feature_size, 1)

    sig=ymin+.5*(sig+1)*(ymax-ymin)
    return sig


def randomLPContour(numSamples, cutoff, sr, rngseed, order=5,  ymin=0, ymax=1) :
    '''
        Create a contour ranging smoothly and randomly between ymin and ymax. 

        @numSamples - of the returned array
        @ cutoff - lp cutoff in Hz
        @sr - sample rate
        @order - of Butterworth LP filter
        @ymin, @ymax -  mapped to this range
        @rngseed - If None, will use random seed 
    '''
    rng = np.random.default_rng(rngseed)

    # start with random noise
    rawg=rng.random(size=numSamples+sr) #zero mean gaussian noise, extra second to 'warm up' lp filter
    # lp filter to smooth the motion
    critical_cutoff_val=10.
    if cutoff < critical_cutoff_val :
        print('randomLPContour: Just so ya know, cutoff below critical cutoff value of {critical_cutoff_val} Hz, using resampling hack to get proper lower frequency behavior')
        rsampfactor = critical_cutoff_val/cutoff
        #templped = butter_lowpass_filter(rawg, critical_cutoff_val, sr, order)[-numSamples:]
        templped = butter_filter(rawg, critical_cutoff_val, sr, 'low', order)[-numSamples:]
        sample=np.linspace(0,int(numSamples/rsampfactor), numSamples)
        lped=np.interp(sample, np.linspace(0, len(templped), len(templped+1)), templped) 
    else :
        #lped = butter_lowpass_filter(rawg, cutoff, sr, order)[-numSamples:]
        lped = butter_filter(rawg, cutoff, sr, 'low', order)[-numSamples:]
    #now map to desired range
    lpedmin=np.amin(lped)
    lpedmax=np.amax(lped)
    mapped = ymin+np.divide(lped-lpedmin,lpedmax-lpedmin)*(ymax-ymin)
    print(f'ymax i {ymax}, and mapped.min is {np.amin(mapped)} and mapped.max is {np.amax(mapped)}')
    return mapped
 

def butter_filter(data, cutoff, sr, btype, order=5):
    ''' @data - array
        @cuttoff - in hz
        @sr
        @btype - {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        @order - default 5
    '''
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)



def map(x, a, b, m, n, clipped=True):
    scaled = float(x - a) / float(b - a)
    extmap = m + (scaled * (n - m))
    if clipped :
        return min(n, max(m,extmap))
    else :
        return extmap

#########################################
# the number of samples returned will be less than the length of snd 
def mvsnd(snd,distance,sr) :
    ''' 
    Takes a sound and a distance (meters) array and returns a new sound array 
    with doppler and amplitude shifts. A distance of 0 will produce no amplitude shift, and amplitude falls off with the square of the distance.
    '''
    #get velocity in meters my taking the difference between two successive points
    rvel=np.diff(-distance, prepend=distance[0])*sr/330 # meters per second
    # create array of real-valued indices for sampling sound (takes bigger steps for higher velocity)
    sample = np.cumsum(rvel+1) 
    #if rvel is negative, shift samples so we always start reading at the beginning of the snd
    if sample[0] < 0 :  
        sample=sample-sample[0]
    if sample[-1] > len(snd) :
        print(f'WARNING: numsamples in snd is {len(snd)}, and the last doppler shifted sample we require is {sample[-1]}. Consider making your snd array a little longer than distance to be sure to have enough for when the average velocity is positive.')
    

    dopplershifted = np.interp(sample, np.linspace(0, len(snd), len(snd+1)), snd) 

    ampscale=1+np.square(distance)
    return np.divide(dopplershifted, ampscale)

#########################################
'''Gestures are transformation function specifying changes about an aspect of a
sound over time. Used for creating amplitude envelopes or frequency sweeps.'''



'''Linearly interpolates from start to stop val
   Startval: Float, int
   Stopval: Float, int
''' 
def gesture(startVal, stopVal, cutOff, numSamples):
        gesture = np.zeros(numSamples)
        non_zero = np.linspace(startVal, stopVal, int(cutOff*numSamples))
        for index in range(len(non_zero)):
                gesture[index] = non_zero[index]
        return gesture

'''Generic gesture creates 2 linear interpolations.'''
''' Startval: Float, int
    Stopval: Float, int 
    2 interpolations: Start to stop, and stop to start
'''
def genericGesture(startVal, stopVal, cutOff, numSamples):
        gesture = np.zeros(numSamples)
        ascending = np.linspace(startVal, stopVal, int(cutOff*numSamples))
        descending = np.linspace(stopVal, startVal, numSamples - int(cutOff*numSamples))
        
        for index in range(len(ascending)):
            gesture[index] = ascending[index]
        for index in range(len(descending)):
            gesture[index+len(ascending)] = descending[index]

        return gesture

##################################################################
#  For time-varying filters
##################################################################
#  ========================================================
#   AllPass
#               original c++ code by Gerry Beauregard
#  ========================================================
#   Second order lattice all-pass filter, as described in
#   Jon Dattorro's "DSP Music Toolbox Part 1", AES Journal
#   1997 September Volume 45 Number 9.
#
#   This filter is not particularly useful by itself, but
#   it serves as the basis for notch, resonator, cut and
#   boost filters described in the same article.
#
#   The all-pass lattice's transfer function is:
#
#                   B + Y(1+B)zm2 + zm2             
#       A(z)    =   --------------------
#                   1 + Y(1+B)zm1 + Bzm2
#
#       "B"   is "beta"
#       "Y"   is "gamma"
#       "zm1" is "z to the power -1"    
#
#   The implementation of a similar lattice is described
#   in Oppenheim & Schafer "Discrete-Time Signal Processing"
#   on p322.  The resulting difference equations are:
#       
#       e2[n] = y;
#       e1[n] = e2[n] - B et1[n]
#       e0[n] = e1[n] - Y et0[n]
#
#       et2[n] = B e1[n] + et1[n-1]
#       et1[n] = Y e0[n] + et0[n-1]
#       et0[n] = e0[n]
#
#       y[n] = et2[n]
#
#       "et" is e with a tilde on top (as per O&S's notation).
#       
#   For stability, we must have |Y| < 1 and |B| < 1.
#
#   (Note: a very similar lattice can be used to implement
#   an all pole filter filter.  The only difference is that
#   the output y[n] is e0[n].  The all-pole lattice is 
#   very useful for modelling speech production).
#

class AllPass(): 
    def __init__(self, i_Y=0, i_B=0) :
        self.m_et1=0
        self.m_et0=0
        self.m_Y=i_Y
        self.m_B=i_B
    
    def setY(self, i_Y) :
        assert i_Y < 1 and i_Y > -1
        self.m_Y = i_Y
        
    def setB(self, i_B) :
        i_B < 1 and i_B > -1
        self.m_B = i_B
    
    def clear(self) :
        self.m_et1=0
        self.m_et0=0      

    def tick(self, i_X ) :
        e2 = i_X
        e1 = e2 - self.m_B*self.m_et1
        e0 = e1 - self.m_Y*self.m_et0

        et2   = self.m_et1 + self.m_B*e1
        self.m_et1 = self.m_et0 + self.m_Y*e0
        self.m_et0 = e0

        return et2

#  ========================================================
#   Resonator.cpp
#  ========================================================
#   2nd order resonator with independent control of 
#   center frequency and Q.  From Jon Dattorro's "DSP Music 
#   Toolbox Part 1", AES Journal 1997 September Vol 45 No 9.
#
#   The resonator uses the 2nd order all-pass filter
#   described in the same article.
#
#   These are the filter's external parameters:
#
#       Fc      Center frequency (Hertz)
#       Q       half-power excursion relative width
#
#   The transfer function for the resonator is
#   
#       H(z)    =   1/2 (1 - A(z))
#
#   Filling in the all-pole transfer function A(z) gives
#
#                      1/2(1-B)(1-zm2)
#       H(z)    =   --------------------
#                   1 + Y(1+B)zm1 + Bzm2
#
#   Read "B"   as "beta"
#        "Y"   as "gamma"
#        "zm1" as "z to the power -1"   
#
#   Y and B are computed as follows:
#   
#       Y   =   -cos(wc)
#   
#               1 - tan(wc/2Q)
#       B   =   --------------
#               1 + tan(wc/2Q)
#
#   where wc is the radian frequency (2*pi*Fc/SR)
#   
class Resonator() :
    '''
    2nd order resonator with independent control of 
    center frequency and Q.  From Jon Dattorro's "DSP Music 
    Toolbox Part 1", AES Journal 1997 September Vol 45 No 9.
    
    The resonator uses the 2nd order all-pass filter described in the same article (above).
    '''
    def __init__(self, i_SR, i_Gain = 1, i_Fc = 440, i_Q = 10) :
        self.m_SR = i_SR
        self.m_Gain = i_Gain   # Gain at center freq (dB)
        self.m_Fc = i_Fc    #Center frequency (Hertz)
        self.m_Q  = i_Q    # FC/BW
        self.m_K  = i_Gain   # Linear gain at center frequency
        self.m_AllPass = AllPass()  #All pass lattice filter
        
        self.set(i_Gain, i_Fc, i_Q)
        self.m_AllPass.clear();
        
    def set(self, i_Gain, i_Fc, i_Q ) :
        assert i_Fc > 0   # Peak cannot be at DC
        assert  i_Fc < self.m_SR/2  # Must be less than the Nyquist frequency!
        assert i_Q > 2*i_Fc/self.m_SR, f'{i_Q=} and {2*i_Fc/self.m_SR=}'# Minimum Q required for filter stability

        self.m_Gain = i_Gain;
        self.m_Fc   = i_Fc;
        self.m_Q    = i_Q;

        # Convert dB gain to absolute linear gain
        self.m_K = i_Gain
        wc = 2*np.pi*self.m_Fc/self.m_SR
        b = (1-np.tan(wc/(2*self.m_Q))) / (1+np.tan(wc/(2*self.m_Q)))
        y = -np.cos(wc)

        self.m_AllPass.setB(b)
        self.m_AllPass.setY(y)
        
    def tick(self, i_X ) :
        x = i_X;
        a = self.m_AllPass.tick(x);
        y = self.m_K * 0.5 * (x-a);

        return y;

# --------------------
# for your convenience  - provide arrays with the sample-by-sample filter values
# sig, g, fc, q all arrays of the same length
def tvBPfilter(sr, sig, g, fc, q) :
    reson=Resonator(sr)
    outsig=np.zeros((len(sig)))
    
    # update the filter coefs on every sample
    for i in range(len(sig)) :
        reson.set(g[i], fc[i], q[i])
        outsig[i]= reson.tick(sig[i])
    return outsig
