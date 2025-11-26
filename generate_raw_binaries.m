% generate_raw_binaries.m
% -----------------------------------------------------------
% This script generates plain binary files containing random
% symbol indices (uint8) for use as input to GNU Radio's
% constellation modulator or chunks-to-symbols block.
%
% No modulation list, no PSD, no .mat output.
% -----------------------------------------------------------

clear; rng(0);

% ==== PARAMETERS ====
numFiles    = 10;       % number of binary files to generate
numSymbols  = 1024;     % number of symbols per file
M           = 16;       % symbol alphabet size (0 to M-1)
outDir      = 'raw_bins';  % output folder name

% Create output folder if needed
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% ==== GENERATE FILES ====
for k = 1:numFiles
    % random symbols in range [0, M-1]
    symIdx = randi([0 M-1], numSymbols, 1, 'uint8');

    % write binary file
    fname = fullfile(outDir, sprintf('symbols_%02d.bin', k));
    fid = fopen(fname, 'w');
    if fid < 0
        error('Cannot open file %s for writing.', fname);
    end
    fwrite(fid, symIdx, 'uint8');
    fclose(fid);
end

fprintf('Generated %d binary files in folder "%s".\n', numFiles, outDir);
fprintf('Each file contains %d uint8 symbols (range 0..%d).\n', numSymbols, M-1);
