%% visqol_evaluation.m
% Matching 1-a-1: ogni originale ha un solo degradato.
% Il degradato contiene nei primi 5 caratteri del suo nome (case-insensitive) la chiave dell'originale.
% Output Excel: col1 = nome file degradato, col2 = MOS-LQO ViSQOL.

clear; clc;

% === Parametri ===
origDir = fullfile(pwd, 'FMA');
degDir  = fullfile(pwd, 'audioseal/FMA');
outXlsx = fullfile(pwd, ['risultati_visqol_audioseal_FMA.xlsx']);
mode = 'audio';   % 'audio' (48 kHz) oppure 'speech' (16 kHz)

audioExt = {'.wav','.flac','.mp3','.m4a','.aac','.ogg','.wma'};

% --- Liste file ---
origList = list_audio_files(origDir, audioExt);
degList  = list_audio_files(degDir,  audioExt);

fprintf('Originali: %d | Degradati: %d\n', numel(origList), numel(degList));

% Prepara nomi base (senza estensione) e versioni lower
degBase  = cell(numel(degList),1);
for d = 1:numel(degList)
    [~, nm, ~] = fileparts(degList(d).name);
    degBase{d} = nm;
end
degBaseL = lower(degBase);
usedDeg  = false(numel(degList),1);   % per evitare riutilizzi

% === Risultati ===
res_names  = {};
res_scores = [];

for i = 1:numel(origList)
    [~, origBase, ~] = fileparts(origList(i).name);
    key = lower(origBase(1:min(5, numel(origBase))));   % primi 5 caratteri

    % trova candidati che contengono la chiave
    candIdx = find(contains(degBaseL, key));

    % escludi degradati già assegnati
    candIdx = candIdx(~usedDeg(candIdx));

    if isempty(candIdx)
        warning('Nessun degradato per "%s" (chiave "%s").', origList(i).name, key);
        continue;
    end

    % Se più candidati, scegli quello "migliore" in modo deterministico:
    % priorità: (1) inizia con chiave, (2) posizione della prima occorrenza, (3) nome più corto
    starts = false(numel(candIdx),1);
    pos    = inf(numel(candIdx),1);
    len    = zeros(numel(candIdx),1);
    for k = 1:numel(candIdx)
        db = degBaseL{candIdx(k)};
        starts(k) = strncmp(db, key, length(key));
        p = strfind(db, key);
        if ~isempty(p)
            pos(k) = p(1);           % <-- niente ifelse: evita p(1) su vettore vuoto
        end
        len(k) = length(db);
    end
    % Ordina: inizio-chiave prima, poi pos più piccola, poi nome più corto
    [~, order] = sortrows([~starts, pos, len], [1 2 3]);   % ~starts: true(=0) prima
    chosen = candIdx(order(1));

    % carica segnali e calcola ViSQOL
    refPath = fullfile(origDir, origList(i).name);
    degPath = fullfile(degDir,  degList(chosen).name);

    try
        [ref, fsRef] = audioread(refPath);
        [deg, fsDeg] = audioread(degPath);

        mos = compute_visqol_score(ref, fsRef, deg, fsDeg, mode);

        res_names{end+1,1}  = degList(chosen).name; %#ok<SAGROW>
        res_scores(end+1,1) = mos;                  %#ok<SAGROW>

        usedDeg(chosen) = true;

        fprintf('OK: %-30s <---> %-30s | ViSQOL=%.3f\n', ...
            origList(i).name, degList(chosen).name, mos);

    catch ME
        warning('Errore su coppia %s / %s: %s', origList(i).name, degList(chosen).name, ME.message);
    end
end

% --- Salva Excel ---
T = table(res_names, res_scores, 'VariableNames', {'file','visqol'});
writetable(T, outXlsx, 'FileType','spreadsheet');
fprintf('Salvato: %s | Coppie valutate: %d\n', outXlsx, height(T));

%% ===== Funzioni locali =====
function files = list_audio_files(folder, exts)
    files = dir(folder);
    files = files(~[files.isdir]);                  % solo file
    keep  = false(numel(files),1);
    for k = 1:numel(files)
        [~,~,e] = fileparts(files(k).name);
        keep(k) = any(strcmpi(e, exts));
    end
    files = files(keep);
end

function mos = compute_visqol_score(ref, fsRef, deg, fsDeg, mode)
    % target Fs
    if strcmpi(mode,'speech')
        targetFs = 16000;
    else
        mode = 'audio';       % fallback
        targetFs = 48000;
    end

    % resampling
    if fsRef ~= targetFs, ref = resample(ref, targetFs, fsRef); end
    if fsDeg ~= targetFs, deg = resample(deg, targetFs, fsDeg); end

    % downmix mono
    if size(ref,2) > 1, ref = mean(ref,2); end
    if size(deg,2) > 1, deg = mean(deg,2); end

    % allinea lunghezze
    L = min(numel(ref), numel(deg));
    ref = ref(1:L); deg = deg(1:L);

    % ViSQOL
    [mos, ~] = visqol(deg, ref, targetFs, 'Mode', mode);
end

