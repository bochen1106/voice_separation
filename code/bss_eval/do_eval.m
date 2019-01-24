clear;
clc;


path_data = '../../../data/DSD100/set_002';
path_result = '../../exp/001/result';

path_vocal = fullfile(path_data, 'audio', 'vocal');
path_accom = fullfile(path_data, 'audio', 'accom');

%%
filename_list = fullfile(path_data, 'h5', 'names_valid.txt');
fid = fopen(filename_list);
names = textscan(fid, '%s');
names = names{1};
fclose(fid);

%%
n = length(names);
SDR_all = zeros(n, 4);


for i = 1 : n
    fprintf('%d of %d\n', i, n);
    name = names{i};
    
    filename_vocal = fullfile(path_vocal, [name, '.wav']);
    filename_accom = fullfile(path_accom, [name, '.wav']);
    wav_vocal = audioread(filename_vocal);
    wav_accom = audioread(filename_accom);
    
    wav_gt = [wav_vocal'; wav_accom'];
    
    wav_gt_mix = wav_vocal + wav_accom;
    wav_gt_mix = [wav_gt_mix'; wav_gt_mix'];
    
    filename_est_1 = fullfile(path_result, [name, '_1.wav']);
    filename_est_2 = fullfile(path_result, [name, '_2.wav']);
    wav_est_1 = audioread(filename_est_1);
    wav_est_2 = audioread(filename_est_2);
    wav_est = [wav_est_1'; wav_est_2'];
    wav_est_mix = wav_est_1 + wav_est_2;
    wav_est_mix = [wav_est_mix'; wav_est_mix'];

    %%
    [SDR, SIR, SAR, perm]=bss_eval_sources(wav_est, wav_gt);
    SDR = SDR(perm);
    [SDR0, SIR0, SAR0, perm0]=bss_eval_sources(wav_est_mix, wav_gt);
    SDR0 = SDR0(perm0);
    SDR_delta = SDR - SDR0;

    SDR_all(i, 1:2) = SDR';
    SDR_all(i, 3:4) = SDR_delta';
    
end

%%

filename_result = fullfile(path_result, 'SDR_matlab.txt');
dlmwrite(filename_result, SDR_all, '\t');






