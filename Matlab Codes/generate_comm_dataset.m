function generate_comm_dataset()
  
  rng(42, 'twister'); 

  codes = {'hamming74','conv12','rs75','turbo13','ldpc34'};
  mods  = {'BPSK','QPSK','QAM16','QAM64'};
  snrList = [-5, 0, 5];
  numFrames =1000;
  frameSymbols = 1024; 

  outFile = fullfile(pwd, 'dataset_all1.mat');

  fprintf('Generating dataset to single file: %s\n', outFile);
  fprintf('Codes: %s\n', strjoin(codes, ', '));
  fprintf('Mods: %s\n', strjoin(mods, ', '));
  fprintf('SNRs: %s dB (Es/N0)\n', strjoin(string(snrList), ', '));
  fprintf('Frames per class: %d, Symbols per frame: %d\n', numFrames, frameSymbols);

 
  encoders = buildEncoders();

  
  C = numel(codes); M = numel(mods); S = numel(snrList);
  totalComb = C * M * S;
  totalFrames = totalComb * numFrames;
  X = zeros(totalFrames, frameSymbols, 2, 'single');
  codeIdx = zeros(totalFrames,1,'uint16');
  modIdx  = zeros(totalFrames,1,'uint16');
  snrIdx  = zeros(totalFrames,1,'uint16');
  snr_dB  = zeros(totalFrames,1,'single');
  writeIdx = 0;


  combIdx = 0;
  tStart = tic;
  for ci = 1:C
    codeName = codes{ci};
    enc = encoders.(codeName);
    rate = enc.rate; 

    for mi = 1:M
      modName = mods{mi};
      bitsPerSym = modBitsPerSymbol(modName);

     
      codedBitsNeeded = frameSymbols * bitsPerSym;

      for si = 1:S
        SNRdB = snrList(si);
        combIdx = combIdx + 1;
        fprintf('[%3d/%3d] Generating %s | %s | SNR=%+d dB ...\n', combIdx, totalComb, upper(codeName), modName, SNRdB);

        
        for n = 1:numFrames
          writeIdx = writeIdx + 1;
          uBits = genInfoBitsForEncoder(enc, codedBitsNeeded);

       
          cBits = applyEncoder(enc, uBits);

          
          tx = modulateBits(cBits, modName);

      
          rx = addAwgnEsNo(tx, SNRdB);

          
          rx = rx(1:frameSymbols);

          
          X(writeIdx,:,1) = single(real(rx));
          X(writeIdx,:,2) = single(imag(rx));
          codeIdx(writeIdx) = ci;
          modIdx(writeIdx)  = mi;
          snrIdx(writeIdx)  = si;
          snr_dB(writeIdx)  = single(SNRdB);
        end
      end
    end
  end
 
  meta = struct();
  meta.codes = codes;
  meta.modulations = mods;
  meta.snrList_dB = snrList;
  meta.framesPerClass = numFrames;
  meta.frameLen = frameSymbols;
  meta.assumptions = [ ...
    "AWGN only; no fading/impairments", ...
    "Es/N0 interpretation for SNR", ...
    "Gray mapping; unit average power", ...
    "No pulse shaping; 1 sample/symbol", ...
    "Encoded length exceeds need; symbols truncated to 1024", ...
    "Order: code -> modulation -> SNR -> frame" ...
  ];

  save(outFile, 'X', 'codeIdx', 'modIdx', 'snrIdx', 'snr_dB', 'codes', 'mods', 'snrList', 'meta', '-v7.3');
  fprintf('Done in %.1f minutes.\n', toc(tStart)/60);
end


function encoders = buildEncoders()
  encoders = struct();

 
  encoders.hamming74.name = 'hamming74';
  encoders.hamming74.rate = 4/7;
  encoders.hamming74.type = 'block';
  encoders.hamming74.blockInfoBits = 4;
  encoders.hamming74.blockCodedBits = 7;
  encoders.hamming74.encodeFcn = @hamming74_encode;


  mustHave('poly2trellis', 'Communications Toolbox (poly2trellis) required for convolutional encoder.');
  trellis = poly2trellis(7, [171 133]);
  encoders.conv12.name = 'conv12';
  encoders.conv12.rate = 1/2;
  encoders.conv12.type = 'conv';
  encoders.conv12.trellis = trellis;
  encoders.conv12.encodeFcn = @(bits) convenc(bits(:).', trellis).';
  mustHave('convenc', 'Communications Toolbox (convenc) required for convolutional encoder.');


  mustHave('comm.RSEncoder', 'Communications Toolbox (comm.RSEncoder) required for RS encoder.');
  encoders.rs75.name = 'rs75';
  encoders.rs75.rate = 5/7;
  encoders.rs75.type = 'block';
  encoders.rs75.blockInfoBits = 5*3;  
  encoders.rs75.blockCodedBits = 7*3; 
  encoders.rs75.encodeFcn = @(bits) rs75_block_encode(bits);

 
  mustHave('comm.TurboEncoder', 'Communications Toolbox (comm.TurboEncoder) required for Turbo encoder.');
  rscTrellis = poly2trellis(4, [15 13], 13); % octal
  Kt = 2048; % interleaver length
  interleaverIdx = randperm(Kt).';
  turboEnc = comm.TurboEncoder('TrellisStructure', rscTrellis, 'InterleaverIndices', interleaverIdx);
  encoders.turbo13.name = 'turbo13';
  encoders.turbo13.rate = 1/3;
  encoders.turbo13.type = 'turbo';
  encoders.turbo13.blockInfoBits = Kt;
  encoders.turbo13.encodeFcn = @(bits) turbo_block_encode(turboEnc, bits, Kt);

  
  mustHave('comm.LDPCEncoder', 'Communications Toolbox (comm.LDPCEncoder) required for LDPC encoder.');
  H = [];
  if exist('wlanLDPCParityCheckMatrix','file') == 2 || exist('wlanLDPCParityCheckMatrix','builtin') == 5
    try
      H = wlanLDPCParityCheckMatrix(3/4, 1296);
    catch
      
    end
  end
  if isempty(H)
    if exist('dvbs2ldpc','file') == 2 || exist('dvbs2ldpc','builtin') == 5
      try
        H = dvbs2ldpc('3/4');
      catch
        try
          H = dvbs2ldpc('3/4','short');
        catch
          try
            H = dvbs2ldpc('3/4','long');
          catch
            % empty
          end
        end
      end
    end
  end
  if ~isempty(H)
    ldpcEnc = comm.LDPCEncoder(H);
    N = size(H,2); K = N - size(H,1);
    encoders.ldpc34.name = 'ldpc34';
    encoders.ldpc34.rate = 3/4;
    encoders.ldpc34.type = 'ldpc';
    encoders.ldpc34.N = N; encoders.ldpc34.K = K;
    encoders.ldpc34.encodeFcn = @(bits) ldpc_encode_blocks(ldpcEnc, K, N, bits);
  else

    K = 768; M = 256; N = K + M;
    wc = 3; 
    P = sparse(M, K);
    rng(7);
    for col = 1:K
      rows = randperm(M, wc);
      P(rows, col) = 1;
    end
 
    emptyRows = find(sum(P,2) == 0);
    for r = emptyRows.'
      cols = randperm(K, wc);
      P(r, cols) = 1;
    end
    encoders.ldpc34.name = 'ldpc34';
    encoders.ldpc34.rate = 3/4;
    encoders.ldpc34.type = 'ldpc';
    encoders.ldpc34.N = N; encoders.ldpc34.K = K; encoders.ldpc34.P = P;
    encoders.ldpc34.encodeFcn = @(bits) ldpc_custom_encode_blocks(bits, K, N, P);
  end
end


function uBits = genInfoBitsForEncoder(enc, codedBitsNeeded)
  switch enc.name
    case 'hamming74'
      blocks = ceil(codedBitsNeeded / enc.blockCodedBits);
      numInfo = blocks * enc.blockInfoBits;
      uBits = randi([0 1], numInfo, 1, 'uint8');

    case 'conv12'
      numInfo = ceil(codedBitsNeeded * enc.rate); 
      uBits = randi([0 1], numInfo, 1, 'uint8');

    case 'rs75'
      blocks = ceil(codedBitsNeeded / enc.blockCodedBits);
      numInfo = blocks * enc.blockInfoBits;
      uBits = randi([0 1], numInfo, 1, 'uint8');

    case 'turbo13'

      blocks = ceil((codedBitsNeeded * enc.rate) / enc.blockInfoBits); 
      numInfo = max(1, blocks) * enc.blockInfoBits;
      uBits = randi([0 1], numInfo, 1, 'uint8');

    case 'ldpc34'
   
      blocks = max(1, ceil(double(codedBitsNeeded) / double(enc.N)));
      numInfo = blocks * enc.K;
      uBits = randi([0 1], numInfo, 1, 'uint8');

    otherwise
      error('Unknown encoder %s', enc.name);
  end
end


function cBits = applyEncoder(enc, uBits)
  switch enc.name
    case 'hamming74'
      cBits = enc.encodeFcn(uBits);
    case 'conv12'
      cBits = enc.encodeFcn(uBits);
      cBits = cBits(:);
    case 'rs75'
    
      cBits = enc.encodeFcn(uBits(:));
    case 'turbo13'
      cBits = enc.encodeFcn(uBits);
    case 'ldpc34'
      cBits = enc.encodeFcn(uBits);
    otherwise
      error('Unknown encoder %s', enc.name);
  end
  cBits = logical(cBits(:));
end

function cBits = hamming74_encode(uBits)
  uBits = logical(uBits(:));
  L = numel(uBits);
  if mod(L,4) ~= 0
    error('uBits length for Hamming(7,4) must be multiple of 4 (provided %d).', L);
  end
  M = L/4;
  u = reshape(uBits, 4, M).'; % [d1 d2 d3 d4] per row
  d1 = u(:,1); d2 = u(:,2); d3 = u(:,3); d4 = u(:,4);
  p1 = xor(xor(d1,d2), d4);
  p2 = xor(xor(d1,d3), d4);
  p4 = xor(xor(d2,d3), d4);
  code = [p1 p2 d1 p4 d2 d3 d4].';
  cBits = code(:);
end

function cBits = turbo_block_encode(turboEnc, bits, Kt)
  bits = logical(bits(:));
  L = numel(bits);
  if mod(L, Kt) ~= 0
    error('Turbo encoder input must be multiple of Kt=%d. Got %d.', Kt, L);
  end
  blocks = L / Kt;
  cBits = false(3*L + 12*blocks, 1); 
  cAll = cell(blocks,1);
  for i=1:blocks
    seg = bits((i-1)*Kt+1:i*Kt);
    cAll{i} = step(turboEnc, seg);
  end
  cBits = vertcat(cAll{:});
end

function cBits = rs75_block_encode(bits)
  bits = uint8(bits(:));
  rsEnc = comm.RSEncoder('CodewordLength',7,'MessageLength',5,'BitInput',true);
  cBits = step(rsEnc, bits);
end

function cBits = ldpc_encode_blocks(ldpcEnc, K, N, bits)
  bits = logical(bits(:));
  L = numel(bits);
  if mod(L, K) ~= 0
    error('LDPC input length must be multiple of K=%d. Got %d.', K, L);
  end
  blocks = L / K;
  cAll = cell(blocks,1);
  for i=1:blocks
    seg = bits((i-1)*K+1:i*K);
    cAll{i} = step(ldpcEnc, seg);
  end
  cBits = vertcat(cAll{:});
  if mod(numel(cBits), N) ~= 0
    cBits = cBits(1:floor(numel(cBits)/N)*N);
  end
end

function cBits = ldpc_custom_encode_blocks(bits, K, N, P)
  bits = logical(bits(:));
  L = numel(bits);
  if mod(L, K) ~= 0
    error('LDPC input length must be multiple of K=%d. Got %d.', K, L);
  end
  blocks = L / K;
  cAll = cell(blocks,1);
  for i=1:blocks
    u = bits((i-1)*K+1:i*K);
    p = mod(P * double(u), 2) > 0;
    cAll{i} = [u; p];
  end
  cBits = vertcat(cAll{:});
  if numel(cBits) ~= blocks * N
    cBits = cBits(1:blocks*N);
  end
end

function M = modBitsPerSymbol(modName)
  switch upper(modName)
    case 'BPSK',  M = 1;
    case 'QPSK',  M = 2;
    case 'QAM16', M = 4;
    case 'QAM64', M = 6;
    otherwise, error('Unknown modulation %s', modName);
  end
end

function x = modulateBits(bits, modName)
  bits = logical(bits(:));
  switch upper(modName)
    case 'BPSK'
      x = 1 - 2*double(bits);
      x = complex(x, 0);

    case 'QPSK'
  k = 2;
  L = floor(numel(bits)/k)*k;
  b = reshape(bits(1:L), 2, []).';
      I = 1 - 2*double(b(:,2));
      Q = 1 - 2*double(b(:,1)); 
      x = (I + 1j*Q) / sqrt(2);

    case {'QAM16','QAM64'}
      M = 2^modBitsPerSymbol(modName);
  mustHave('qammod', 'Communications Toolbox (qammod) required for QAM.');
      k = log2(M);
  L = floor(numel(bits)/k)*k;
  x = qammod(uint8(bits(1:L)), M, 'InputType','bit', 'UnitAveragePower', true);

    otherwise
      error('Unknown modulation %s', modName);
  end
end

function y = addAwgnEsNo(x, SNRdB)
  x = x(:);
  p = mean(abs(x).^2);
  if p <= 0
    y = x; return;
  end
  x = x / sqrt(p);
  SNRlin = 10.^(SNRdB/10);
  noiseVar = 1 ./ SNRlin; % Es=1
  n = sqrt(noiseVar/2) * (randn(size(x)) + 1j*randn(size(x)));
  y = x + n;
end

function mustHave(nameOrFcn, msg)
  if isa(nameOrFcn, 'function_handle')
    name = func2str(nameOrFcn);
  else
    name = char(string(nameOrFcn));
  end
  ok = (exist(name,'builtin')==5) || (exist(name,'file')==2) || (exist(name,'class')==8) || (exist(name,'mex')==3) || (exist(name,'pcode')==6);
  if ~ok
    error('%s (missing: %s)', msg, name);
  end
end
