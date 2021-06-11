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
par.config.target_dir_deg = 15;%frontal target (0:5:355)
par.config.look_dir_deg = 0;% (0:5:355)
par.config.snr_db_ref_left = +5; %
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

% test: soundsc([all_clean_mic_sigs(:,1) all_clean_mic_sigs(:,3)],par.sim.fs)
%       should come from specified direction (par.config.target_dir_deg)
%       when listening in headphones.

% now create noisy signal
all_noisy_mic_sigs = all_clean_mic_sigs + all_noise_mic_sigs;

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
    S_mat = stft(par.stft,clean_mic_sigs);
    V_mat = stft(par.stft,noise_mic_sigs);
    X_mat = S_mat + V_mat; %The STFT is linear so we can sum STFTs
    
    [numBands, numFrames, M] = size(S_mat);
     %% Initialize my variables            
    twoSecFrames = floor(2 * par.sim.fs * numFrames / length(all_noisy_mic_sigs));
    twoSecNoise_mat = X_mat(:, 1:twoSecFrames, :);
    D = 20;
%     X_XH_sum_noiseOnly = 0;
%     X_XH_sum = 0;
    beta = 50;
    IVAD_threshold = 25;
    
     %% estimate Gamma V = normalized noise CPSD matrix with respect to the reff microphone
    for iBand = 1:numBands
        for iFrame = 1:twoSecFrames
            if iFrame < twoSecFrames
               X_XH_sum_noiseOnly(:, :, iBand) = squeeze(X_mat(iBand, iFrame, :)) * squeeze(X_mat(iBand, iFrame, :))' / twoSecFrames; 
            end
        end
        %Cv_hat(:, :, iBand) = X_XH_sum_noiseOnly(:, :, iBand);
        Cv_hat(:, :, iBand) = squeeze(sum(X_XH_sum_noiseOnly(:, :, iBand), [1 2]));
        gamma_V(:, :, iBand) = Cv_hat( :, :, iBand) / Cv_hat(ii_ref, ii_ref, iBand);
    end

    
    %% Processing
    % do beamforming for all freq. bands and all time frames.
    % the implementation below is inefficient - but hopefully easy to read
    for iBand = 1:numBands
        d_kl = d_vecs(:,iBand);%get relative transfer function
        X_l = squeeze(X_mat(iBand,:,:)).';%mic sigs as cols for this freq
        for iFrame = D:numFrames

            X_kl = X_l(:,iFrame);%Mx1 noisy mic. sig.
            
            %% CODE TO BE FILLED IN BY COURSE PARTICIPANTS
            %  ... 
            X = X_l(:, max(iFrame - D + 1, 1):iFrame);
            Cx_hat = (X * X') / D;
            Cx_hat_inv = inv(Cx_hat);
            [d_ml, lambda_s_ml, lambda_v_ml] = ml_known_covariance_structure_fun(Cx_hat, gamma_V(:,:,iBand), ii_ref);
            Cv = lambda_v_ml * gamma_V(:,:,iBand);
            
            if iFrame > D
                %DSB
                W_DSB(:, iBand,iFrame) = d_kl/(d_kl' * d_kl); %DSB Beamformer
                X_W_DSB(iBand, iFrame) = W_DSB(:, iBand, iFrame)' * X_kl; %beamformer output

                %MVDR
                W_MVDR(:, iBand, iFrame) = (Cx_hat_inv * d_kl) / (d_kl' * Cx_hat_inv * d_kl); %version one
                %W_MVDR(:, iBand, iFrame) = (inv(Cv) * d_kl)/(d_kl' * inv(Cv) * d_kl); %version 2
                X_W_MVDR(iBand, iFrame) = W_MVDR(:, iBand, iFrame)' * X_kl;

                %MVDR with unknown d
                W_MVDR_unknown(:, iBand, iFrame) = (Cx_hat_inv * d_ml) / (d_ml' * Cx_hat_inv * d_ml);
                X_W_MVDR_unknown(iBand, iFrame) = W_MVDR_unknown(:, iBand, iFrame)' * X_kl;

                %MWF
                Cx = lambda_s_ml * (d_kl * d_kl') + lambda_v_ml * gamma_V(:,:,iBand);
                Cs = Cx - Cv;
                e = zeros(length(par.config.i_mics_left), 1);
                e(ii_ref) = 1;
                W_MWF(:, iBand, iFrame) = inv(Cx)*Cs*e;
                X_W_MWF(iBand, iFrame) = W_MWF(:, iBand, iFrame)' * X_kl;
            else
                 X_W_DSB(iBand, iFrame) = X_kl(ii_ref);
                 X_W_MVDR(iBand, iFrame) = X_kl(ii_ref);
                 X_W_MVDR_unknown(iBand, iFrame) = X_kl(ii_ref);
                 X_W_MWF(iBand, iFrame) = X_kl(ii_ref);
            end
            
        end
    end
    
    %% Plot Graphs
    tiledlayout(4,1);

    nexttile;
    imagesc(mag2db(abs(X_W_DSB)));
    colormap(summer);
    set(gca, 'YDir', 'normal'); 
    title('Result of DSB beamformer');
    
    nexttile;
    imagesc(mag2db(abs(X_W_MVDR)));
    colormap(summer);
    set(gca, 'YDir', 'normal'); 
    title('Result of MVDR beamformer');
    
    nexttile;
    imagesc(mag2db(abs(X_W_MVDR_unknown)));
    colormap(summer);
    set(gca, 'YDir', 'normal'); 
    title('Result of MVDR beamformer with unknown RATF');
    
    nexttile;
    imagesc(mag2db(abs(X_W_MWF)));
    colormap(summer);
    set(gca, 'YDir', 'normal'); 
    title('Result of MWF beamformer');
    %% Create audio files
    s_DSB = Normalize(istft(par.stft,  X_W_DSB) ,1);
    s_MVDR = Normalize(abs(istft(par.stft, X_W_MVDR)) ,1);
    s_MVDR_u = Normalize(abs(istft(par.stft, X_W_MVDR_unknown)) ,1);
    s_MWF = Normalize(abs(istft(par.stft, X_W_MWF)) ,1);
    
    s_MVDR(isnan(s_MVDR)) = 0;
    s_MVDR_u(isnan(s_MVDR_u)) = 0;
    s_MWF(isnan(s_MWF)) = 0;
    
    audiowrite("Audio/DSB.wav", s_DSB, par.sim.fs);
    audiowrite("Audio/MVDR.wav", s_MVDR, par.sim.fs);
    audiowrite("Audio/MVDR_u.wav", s_MVDR_u, par.sim.fs);
    audiowrite("Audio/MWF.wav", s_MWF, par.sim.fs);
  
    %%
    % keep clean and noisy sigs at ref. mic for objective evaluation
    x_ref = all_noisy_mic_sigs(:,i_mics(ii_ref));
    s_ref = all_clean_mic_sigs(:,i_mics(ii_ref));
      
    % convert STFTs to time domain
    % s_mvdr = istft(par.stft,S_mvdr_stft);
    % s_mwf = istft(par.stft, S_mwf_stft);
        
    % save time domain signals for objective evaluation
    x_ref_l = x_ref;
    s_ref_l = s_ref;
    %s_mvdr_l = s_mvdr;
    %s_mwf_l = s_mwf;
    
    %% Perform Analysis
    ESTOI(1) = estoi(s_ref, s_DSB, par.sim.fs);
    ESTOI(2) = estoi(s_ref, s_MVDR, par.sim.fs);
    ESTOI(3) = estoi(s_ref, s_MVDR_u, par.sim.fs);
    ESTOI(4) = estoi(s_ref, s_MWF, par.sim.fs);
    
    DSB.s = s_ref;
    DSB.x = s_DSB;
    
    MVDR.s = s_ref;
    MVDR.x = s_MVDR;
    
    MVDR_u.s = s_ref;
    MVDR_u.x = s_MVDR_u;
    
    MWF.s = s_ref;
    MWF.x = s_MWF;
    
    seg_SNR(1) = seg_snr(DSB);
    seg_SNR(2) = seg_snr(MVDR);
    seg_SNR(3) = seg_snr(MVDR_u);
    seg_SNR(4) = seg_snr(MWF);
    
    figure
    plot (seg_SNR(1).ssnr);
    hold on;
    plot (seg_SNR(2).ssnr);
    hold on;
    plot (seg_SNR(3).ssnr);
    hold on;
    plot (seg_SNR(4).ssnr);
    legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
    title (sprintf('Segmental SNR for input SNR = %d', par.config.snr_db_ref_left));
    
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
