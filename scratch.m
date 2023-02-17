% Load EEG spike train for this network of this run of this subject
eeg = load('/homes/4/greve/tmp/s17_121-r1.non-nan.par');
neeg = length(eeg);
fsample_eeg = 20; % Hz
teeg = [0:neeg-1]'/fsample_eeg;
plot(teeg,eeg)

% Ideal fMRI HRF response
% time sampled at EEG rate at a long enough window (30sec) to
% capture the response
thrf = [0:600]/fsample_eeg; 
% Parameters of the parametric HRF model
delta = 2.25; % sec
tau = 1.25;
alpha = 2;
% Ideal HRF waveform
hrf = fmri_hemodyn(thrf,delta,tau,alpha);
plot(thrf,hrf)

% This is the ideal fMRI waveform assuming these parameters
% sampled at the EEG
hrfeeg = conv(eeg,hrf,'same');
plot(teeg,hrfeeg,teeg,eeg)

% Down-sample to fMRI sample rate
TR_fmri = 800; % msec
r_fmri = round(fsample_eeg*800/1000); % 16 EEG points for each fMRI point
tfmri = teeg(1:r_fmri:end);
hrfeeg_fmri = hrfeeg(1:r_fmri:end);
nfmri = length(tfmri);

plot(tfmri,hrfeeg_fmri,'+-',teeg,eeg)

plot(tfmri,hrfeeg_fmri,'.-',teeg,hrfeeg,teeg,eeg)
legend('HRF-fMRI','HRF-EEG','EEG spikes')

randn('state',53); % force noise to be the same in each iteration

% Simulate a raw fMRI waveform for now
fmri = hrfeeg_fmri + 3*randn(nfmri,1); % 3=noise level
plot(tfmri,hrfeeg_fmri,tfmri,fmri);

% Brute force search of tau over these values
% We know the ideal value is the value of tau above, 
% now let's see if we can find it.
taulist = [1:.01:1.5];
ntau = length(taulist);
rvarlist = zeros(ntau,1);
rvarmin = 10e10;
taumin = 0;
for nthtau = 1:ntau
  % Go through the process again with this tau
  tau_test = taulist(nthtau);
  hrf_test = fmri_hemodyn(thrf,delta,tau_test,alpha);  
  hrfeeg_test = conv(eeg,hrf_test,'same');
  hrfeeg_fmri_test = hrfeeg_test(1:r_fmri:end);
  % Fit this time course to raw fMRI to waveform with a GLM
  % beta = inv(X'*X)*X'*fmri; yhat = X*beta; residual = fmri-yat; residual
  % variance = std(residual). Same effect as doing the correlation
  % but resicual variance is a cost we want to minimize rather than
  % a correlation to maximize
  X = [ones(nfmri,1) hrfeeg_fmri_test];
  [beta_test rvar_test] = fast_glmfit(fmri,X);
  %[F p] = fast_fratio(beta_test,rvar_test,[0 1]);
  if(rvarmin > rvar_test)
    % Keep track of the minimum cost and the corresponding 
    rvarmin = rvar_test;
    taumin = tau_test;
  end
  rvarlist(nthtau) = rvar_test;
end

% Where is the tau of the minimum variance and how close to ideal?
plot(taulist,rvarlist,'+-',taumin,rvarmin,'*')
xlabel('Tau')
ylabel('Fit Cost');
title(sprintf('Tau Min %g, Ideal %g',taumin,tau));

