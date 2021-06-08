clc
clear
% This is a matlab script skeleton for simulation a sound field impinging on a
% hearing aid system (a 2-microphone hearing aid, located at each ear).
%
% Author:
% Jesper Jensen, CASPR, Aalborg University, 2020.

% add path to objective distortion measures
addpath_cmd = ['addpath ' strcat(pwd,filesep,'obj_dist_measures')];
eval(addpath_cmd);

%%
% Initialize
%
disp('Initialize.')
par.sim.micsigDir = 'mic_sigs/';
par.sim.fs = 20e3;
% microphone nomenclature -  1: left, front, 2: left, rear, 3: right,front, 4: right, rear.
par.config.i_mics_left = [1 2];%mics used for left-ear beamformer (leave empty to bypass processing)
par.config.i_ref_left = 1;%index of left reference mic. (left-front)
par.config.i_mics_right = [3 4];%1 2 3 4];%mics used of right-ear beamformer
par.config.i_ref_right = 3;%index of right reference mic. (right-front)
par.config.target_dir_deg = 15;%frontal target (0:5:355) CCW
par.config.look_dir_deg = 0;% (0:5:355) CCW probs not use
par.config.snr_db_ref_left = 5; % SNR at the reff microphone
par.stft.frame_length  = 256;  %corresponds to 12.8ms at fs=20000
par.stft.awin = sqrt(mod_hann(par.stft.frame_length));%use sqrt hann. window
par.stft.swin = sqrt(mod_hann(par.stft.frame_length));%synth.window
par.stft.D_A = par.stft.frame_length/2;
par.stft.N = par.stft.frame_length;%fft order
par.ideal_vad.thr_db = 50; %cheat vad: clean frame energy 50 dB less than max

%%
% Set up an acoustic scene
%
% The basic building blocks are three pre-stored files: mic_sigs_ssn_iso.mat and
% mic_sigs_bbl_iso.mat which contain "cylindrically isotropic" speech shaped 
% noise and babble speech signals (72 directions), and mic_sigs_trg.mat, 
% which contains a single speech target from any of 72 directions (0:5:355 degs).
% To achieve a desired SNR, the isotropic disturbance signal (ssn or bbl) is scaled so
% that the energy of the target signal vs. the energy of the disturbance,
% e.g., at the reference microphone, has the desired SNR.

disp('Set up acoustic scene.')
% make noise signals
noise_mic_sig_filename = 'mic_sigs_bbl_iso.mat';%(choose mic_sigs_bbl_iso or mic_sigs_ssn_iso)

% load microphone signals (noise) for all directions
load_cmd = ['load ' par.sim.micsigDir noise_mic_sig_filename];
eval(load_cmd)

% rename loaded vars.
if strcmp(noise_mic_sig_filename,'mic_sigs_bbl_iso.mat')
    Xo_noi = Xo_bbl_iso;
elseif strcmp(noise_mic_sig_filename,'mic_sigs_ssn_iso.mat')
    Xo_noi = Xo_ssn_iso;
else
    disp('error');keyboard
end


% make isotropic noise (sum all directions) for all mics
all_noise_mic_sigs = [Xo_noi.frontLeft.x Xo_noi.rearLeft.x Xo_noi.frontRight.x Xo_noi.rearRight.x];
 
% test: soundsc([all_noise_mic_sigs(:,1) all_noise_mic_sigs(:,3)],par.sim.fs)
%       Should sound isotropic in headphones.

% make target signal
speech_mic_sig_filename = 'mic_sigs_trg.mat';%
load_cmd = ['load ' par.sim.micsigDir speech_mic_sig_filename];
eval(load_cmd);% load Xo_trg (i.e., targets from all directions)

% generate noisy signals (i.e., scale to desired frontal snr)
snr_ref_dir_deg = 0; %scale for frontal snr
[val, i_trg_dir] = find(round(theta/2/pi*360) == snr_ref_dir_deg);
frontal_clean_mic_sigs = [Xo_trg.frontLeft.x(:,i_trg_dir) Xo_trg.rearLeft.x(:,i_trg_dir) Xo_trg.frontRight.x(:,i_trg_dir) Xo_trg.rearRight.x(:,i_trg_dir)];

% convention: first, we scale all signals so that the SNR at the left reference
% microphone is as desired (as specified in par.config.snr_db_left [dB]) WHEN THE
% TARGET SIGNAL IS FRONTAL. Then we FIX the power of the target
% source and all noise sources. Next, we place the target (without
% changing its power) in the desired direction (as specified in
% par.config.target_dir_deg). Note: a consequence of this is that
% for a target placed in 80 degrees (i.e., left) the SNR at the
% left reference microphone is probably higher than
% par.config.snr_db_left [dB] because the target source experiences
% less head shadow than when the target was frontal. Similarly, for a target placed in 280 degrees
% (i.e., right) the SNR experienced at the left reference
% microphone is going to be lower than par.config.snr_db_left [dB]
% because the target sound is going to pass the head.
% 

noise_ref_mic = all_noise_mic_sigs(:,par.config.i_ref_left);%isotropic noise at ref mic
clean_ref_mic = frontal_clean_mic_sigs(:,par.config.i_ref_left);%frontal speech at ref mic
g_trg = sqrt(var(noise_ref_mic)/var(clean_ref_mic)*10^(par.config.snr_db_ref_left/10));
%ensure desired snr at ref mic for frontal speech
Xo_trg.frontLeft.x = g_trg*Xo_trg.frontLeft.x;
Xo_trg.rearLeft.x = g_trg*Xo_trg.rearLeft.x;
Xo_trg.frontRight.x = g_trg*Xo_trg.frontRight.x;
Xo_trg.rearRight.x = g_trg*Xo_trg.rearRight.x;

% now pick mic.sics for target signal from correct direction
[val, i_trg_dir] = find(round(theta/2/pi*360) == par.config.target_dir_deg);
all_clean_mic_sigs = [Xo_trg.frontLeft.x(:,i_trg_dir) Xo_trg.rearLeft.x(:,i_trg_dir) Xo_trg.frontRight.x(:,i_trg_dir) Xo_trg.rearRight.x(:,i_trg_dir)];

%soundsc([all_clean_mic_sigs(:,1) all_clean_mic_sigs(:,3)],par.sim.fs);
% test: soundsc([all_clean_mic_sigs(:,1) all_clean_mic_sigs(:,3)],par.sim.fs)
%       should come from specified direction (par.config.target_dir_deg)
%       when listening in headphones.

% now create noisy signal
all_noisy_mic_sigs = all_clean_mic_sigs + all_noise_mic_sigs;

%soundsc(all_noisy_mic_sigs(:,par.config.i_ref_left), par.sim.fs);
%%
% Pick out relevant microphone signals and perform beamforming
% (first left, then right)
%

%
% Processing
%
if ~isempty(par.config.i_mics_left)%should we process left ear HA?
    
    % initialize relative acoustic transfer functions wrt. ref.mic.
    disp('processing for left-ear device.')
    i_mics = par.config.i_mics_left;%par.config.i_mics_left/par.config.i_mics_right;
    i_ref = par.config.i_ref_left;%par.config.i_ref_left/par.config.i_ref_right;
    ii_ref = find(i_mics == i_ref);%index of ref mic. in selected mic. set.
    
    look_dir_deg = par.config.look_dir_deg;
    d_vecs = compute_d_vecs(par,look_dir_deg,i_mics,i_ref);

    disp('Perform speech enhancement.')
    clean_mic_sigs = all_clean_mic_sigs(:,i_mics);
    noise_mic_sigs = all_noise_mic_sigs(:,i_mics);
    noisy_mic_sigs = clean_mic_sigs + noise_mic_sigs;
    
    % perform STFT for all signals
    S_mat = stft(par.stft, clean_mic_sigs); % clean singal tile matrix
    S_mat_dB = mag2db(abs(S_mat));
    V_mat = stft(par.stft, noise_mic_sigs); % noise signal tile matrix
    X_mat = S_mat + V_mat; %The STFT is linear so we can sum STFTs
    X_mat_dB = mag2db(abs(X_mat));
    
    [numBands, numFrames, M] = size(S_mat);

    %X_XH = zeros(numBands, twoSecFrames, M, M );
    %Cv_hat = zeros(numBands, M, M);
    %gamma_V = zeros(numBands, M, M);
    %% Initialize my variables
               
    twoSecFrames = floor(2 * par.sim.fs * numFrames / length(all_noisy_mic_sigs));
    twoSecNoise_mat = X_mat(:, 1:twoSecFrames, :);
    D = 20;
    X_XH_sum_noiseOnly = 0;
    X_XH_sum = 0;
    beta = 50;
    IVAD_threshold = 25;
    
    %% estimate Gamma V = normalized noise CPSD matrix with respect to the reff microphone
    for iBand = 1:numBands
        for iFrame = 1:twoSecFrames
            if iFrame < twoSecFrames
               X_XH_sum_noiseOnly = X_XH_sum_noiseOnly + squeeze(X_mat(iBand, iFrame, :)) * squeeze(X_mat(iBand, iFrame, :))';  
            end
        end 
    end
    Cv_hat =  X_XH_sum_noiseOnly / twoSecFrames;
    gamma_V = Cv_hat( :, :) / Cv_hat(ii_ref, ii_ref);

    %uncomment to hear the first 2 sec of noise
    %twoSecNoise = (all_noisy_mic_sigs(1:2 * par.sim.fs, ii_ref)); 
    
    %% 
    % do beamforming for all freq. bands and all time frames.
    % the implementation below is inefficient - but hopefully easy to read
    for iBand = 1:numBands
        d_kl = d_vecs(:,iBand);%get relative transfer function
        X_l = squeeze(X_mat(iBand,:,:)).';%mic sigs as cols for this freq
        for iFrame = 1:numFrames

            X_kl = X_l(:,iFrame);%Mx1 noisy mic. sig.
            
            %% CODE TO BE FILLED IN BY COURSE PARTICIPANTS
            %  ... 
            if iFrame > D
                for l_prime = 1:D
                    X_XH_sum = X_XH_sum + squeeze(X_mat(iBand, iFrame - D + l_prime, :)) * squeeze(X_mat(iBand, iFrame - D + l_prime, :, :))';
                end
            end
            X_XH = squeeze(X_mat(iBand, iFrame, :)) * squeeze(X_mat(iBand, iFrame, :, :))';
            Cx_hat = X_XH_sum / numFrames;
            [lambda_s_ml, lambda_v_ml] = ml_known_cova_struct_and_d_fun(Cx_hat, gamma_V, d_kl, ii_ref);
            logLikelihood(iBand, iFrame) = D * log(det(1/M * trace(X_XH * gamma_V') * gamma_V)) - D * log(det(lambda_s_ml * (d_kl * d_kl') + lambda_v_ml * gamma_V)); %eq 13
            logLikelihood_dB(iBand, iFrame) = mag2db(abs(logLikelihood(iBand, iFrame)));
            
            %log likelihood VAD
            if logLikelihood_dB(iBand, iFrame) > beta
                VAD(iBand, iFrame) = 0;
            else
                VAD(iBand, iFrame) = 1;
            end
            
            %Ideal VAD
            if S_mat_dB(iBand, iFrame) < IVAD_threshold
                IVAD(iBand, iFrame) = 0;
            else
                IVAD(iBand, iFrame) = 1;
            end
            
            %VAD overlap
            if VAD(iBand,iFrame) == IVAD(iBand, iFrame)
                VAD_overlap(iBand, iFrame) = 0;
            else
                VAD_overlap(iBand, iFrame) = 1;
            end
        end
    end
    % keep clean and noisy sigs at ref. mic for objective evaluation
    x_ref = all_noisy_mic_sigs(:,i_mics(ii_ref));
    s_ref = all_clean_mic_sigs(:,i_mics(ii_ref));
    %% Plot graphs
    spec = spectrogram(x_ref, par.stft.awin, 255, [], par.sim.fs, 'yaxis');
    
    tiledlayout(5,1)
    
    nexttile
    spectrogram(x_ref, par.stft.awin, 255, [], par.sim.fs, 'yaxis');
    colormap(summer);
    colorbar( 'off' )
    title('X ref');
    
    nexttile
    imagesc(logLikelihood_dB);
    colormap(summer);
    set(gca, 'YDir', 'normal');
    title('Log likelihood magnitude');
    
    nexttile
    imagesc(VAD);
    colormap(summer);
    set(gca,'YDir','normal');
    title(sprintf('Log Likelihood VAD with threshold = %d', beta));
    
    nexttile
    imagesc(IVAD);
    colormap(summer);
    set(gca, 'YDir', 'normal');
    title(sprintf('Ideal VAD with threshold = %d', IVAD_threshold));
    
    nexttile
    imagesc(VAD_overlap);
    colormap(summer);
    set(gca, 'YDir', 'normal');
    title('Binary verlapping tiles between the LLVAD and IVAD');
    txt = 'Colors have been inverted for readability';
    text(10, 110, txt, 'FontSize', 10);
    
    
    % convert STFTs to time domain
    % s_mvdr = istft(par.stft,S_mvdr_stft);
    % s_mwf = istft(par.stft, S_mwf_stft);
        
    % save time domain signals for objective evaluation
    x_ref_l = x_ref;
    s_ref_l = s_ref;
    %s_mvdr_l = s_mvdr;
    %s_mwf_l = s_mwf;
    
end %endif: should we process left ear HA?


%
% Processing
%
if ~isempty(par.config.i_mics_right)%should we process right ear HA?
    
    % initialize relative acoustic transfer functions wrt. ref.mic.
    disp('processing for right-ear device.')
    i_mics = par.config.i_mics_right;%par.config.i_mics_left/par.config.i_mics_right;
    i_ref = par.config.i_ref_right;%par.config.i_ref_left/par.config.i_ref_right;
    ii_ref = find(i_mics == i_ref);%index of ref mic. in selected mic. set.
    
    look_dir_deg = par.config.look_dir_deg;
    d_vecs = compute_d_vecs(par,look_dir_deg,i_mics,i_ref);
    
    disp('Perform speech enhancement.')
    clean_mic_sigs = all_clean_mic_sigs(:,i_mics);
    noise_mic_sigs = all_noise_mic_sigs(:,i_mics);
    noisy_mic_sigs = clean_mic_sigs + noise_mic_sigs;
    
    % perform STFT for all signals
    S_mat = stft(par.stft,clean_mic_sigs);
    V_mat = stft(par.stft,noise_mic_sigs);
    X_mat = S_mat + V_mat; %The STFT is linear so we can sum STFTs
    
    [numBands, numFrames, M] = size(S_mat);
    
    % do beamforming for all freq. bands and all time frames.
    % the implementation below is inefficient - but hopefully easy to read
    
    W_DSB = zeros(M, numBands, numFrames);
    XW_DSB = zeros(M, numFrames);
    
    for iBand = 1:numBands
        d_kl = d_vecs(:,iBand);%get relative transfer function
        X_l = squeeze(X_mat(iBand,:,:)).';%mic sigs as cols for this freq
        for iFrame = 1:numFrames
            
            X_kl = X_l(:,iFrame);%Mx1 noisy mic. sig.
            
            %% CODE TO BE FILLED IN BY COURSE PARTICIPANTS
            %  ... 
        end
    end
    
    x_ref = all_noisy_mic_sigs(:,i_mics(ii_ref));
    s_ref = all_clean_mic_sigs(:,i_mics(ii_ref));
    
    % convert STFTs to time domain
    % s_mvdr = istft(par.stft,S_mvdr_stft);
    % s_mwf = istft(par.stft, S_mwf_stft);
    
    % save time domain signals for objective evaluation
    x_ref_r = x_ref;
    s_ref_r = s_ref;
    % s_mvdr_r = s_mvdr;
    % s_mwf_r = s_mwf;
    
end %endif: should we process right ear HA?


%%
% compute objective distortion measures
%
% compute output snr left and right device
% snr_in_ref_l = 10*log10(var(s_ref_l)/var(x_ref_l - s_ref_l));
% ...

% compute segmental-snr left and right device
% ssnr_in.s = s_ref_l;
% ssnr_in.x = x_ref_l;
% ssnr_in.fs = par.sim.fs;
% ssnr_out_x_struct = seg_snr(ssnr_in);
%%  seg-snr over speech active frames
% ssnr_x_l = mean(ssnr_out_x_struct.ssnr(ssnr_out_x_struct.vad_index))
% ...

% compute stoi left and right device
% stoi_x_l = stoi(s_ref_l,x_ref_l,par.sim.fs)
% ...

% compute estoi left and right device
% estoi_x_l = estoi(s_ref_l,x_ref_l,par.sim.fs)
% ...

% compute binaural stoi
% dbstoi_x = dbstoi(s_ref_l, s_ref_r, x_ref_l, x_ref_r, par.sim.fs)
% ...

%%
% test: to play back enhanced binaural signals....
% Example soundsc([s_mwf_l(:) s_mwf_r(:)],par.sim.fs)
