function Purity = compute_purity(Y, predY)

if size(Y,2) ~= 1
    Y = Y';
end;
if size(predY,2) ~= 1
    predY = predY';
end;

n = length(Y);

uY = unique(Y);
nclass = length(uY);
Y0 = zeros(n,1);
if nclass ~= max(Y)
    for i = 1:nclass
        Y0(find(Y == uY(i))) = i;
    end;
    Y = Y0;
end;

uY = unique(predY);
nclass = length(uY);
predY0 = zeros(n,1);
if nclass ~= max(predY)
    for i = 1:nclass
        predY0(find(predY == uY(i))) = i;
    end;
    predY = predY0;
end;


% Lidx = unique(Y); classnum = length(Lidx);
predLidx = unique(predY); pred_classnum = length(predLidx);

% purity
correnum = 0;
for ci = 1:pred_classnum
    incluster = Y(find(predY == predLidx(ci)));
%     cnub = unique(incluster);
%     inclunub = 0;
%     for cnubi = 1:length(cnub)
%         inclunub(cnubi) = length(find(incluster == cnub(cnubi)));
%     end;
    inclunub = hist(incluster, 1:max(incluster)); if isempty(inclunub) inclunub=0;end;
    correnum = correnum + max(inclunub);
end;
Purity = correnum/length(predY);