function lap2d_green(nthreads)
% LAP2D_GREEN  Benchmark direct evaluation of the Laplace 2D Green's function.
%
%   G(x,y) = -log(|x-y|^2) / (4*pi)
%
% lap2d_green()         -- single-threaded (default)
% lap2d_green(N)        -- ask MATLAB to use up to N compute threads
%
% Each rep k = 0..NREP-1 shifts the target X coordinates by k*DT and
% adds the resulting NT-by-NS kernel matrix into an accumulator `acc`.
% The final fingerprint (selected entries, min, max, sum) is taken on
% acc, so every rep contributes to the printed answer -- nothing can
% be hoisted out of the timing loop.

if nargin < 1, nthreads = 1; end

NS = 4000;
NT = 4000;
NREP = 20;
DT = 0.01;

actual_nthreads = nthreads;
try
    maxNumCompThreads(nthreads);
    actual_nthreads = maxNumCompThreads;
catch
    % maxNumCompThreads not available; fall through with requested value
end

[src, targ] = make_inputs(NS, NT);

% Warm up (one full pass through the kernel; result discarded).
acc = compute_kernel(src, targ, 0, DT);  %#ok<NASGU>

% Timed loop: accumulate NREP shifted kernel matrices into acc.
acc = zeros(NT, NS);
t0 = tic;
for k = 0:NREP-1
    acc = acc + compute_kernel(src, targ, k, DT);
end
t = toc(t0);

i_mid = NT/2;
j_mid = NS/2;

vmin = min(acc(:));
vmax = max(acc(:));
csum = sum(acc(:));

fprintf('=== MATLAB lap2d_green ===\n');
fprintf('threads        = %d\n', actual_nthreads);
fprintf('NS=%d  NT=%d  NREP=%d  DT=%.4f\n', NS, NT, NREP, DT);
fprintf('elapsed_total  = %.6f s\n', t);
fprintf('elapsed_per    = %.6f s\n', t/NREP);
fprintf('throughput     = %.3e pts/s\n', NS*NT*NREP/t);
fprintf('acc(1,1)       = %.17e\n', acc(1,1));
fprintf('acc(%d,%d) = %.17e\n', i_mid, j_mid, acc(i_mid, j_mid));
fprintf('acc(NT,NS)     = %.17e\n', acc(NT,NS));
fprintf('min(acc)       = %.17e\n', vmin);
fprintf('max(acc)       = %.17e\n', vmax);
fprintf('sum(acc)       = %.17e\n', csum);
end

function val = compute_kernel(src, targ, k, dt)
% NT-by-NS matrix of -log(|t-s|^2)/(4*pi) with target X shifted by k*dt.
rx = (targ(1,:) + k*dt).' - src(1,:);
ry = targ(2,:).' - src(2,:);
r2 = rx.^2 + ry.^2;
val = -log(r2) / (4*pi);
end

function [src, targ] = make_inputs(NS, NT)
% Deterministic xorshift64 PRNG; sources in [-1,1]^2,
% targets in [-1,1]^2 + (3,0) so |t-s| is bounded away from 0.

state = uint64(2463534242);
src  = zeros(2, NS);
targ = zeros(2, NT);

for j = 1:NS
    [u, state] = xs64(state); src(1,j) = u*2 - 1;
    [u, state] = xs64(state); src(2,j) = u*2 - 1;
end
for j = 1:NT
    [u, state] = xs64(state); targ(1,j) = u*2 - 1 + 3;
    [u, state] = xs64(state); targ(2,j) = u*2 - 1;
end
end

function [u, state] = xs64(state)
% xorshift64 -> uniform double in [0,1)
x = state;
x = bitxor(x, bitshift(x, 13));
x = bitxor(x, bitshift(x, -7));
x = bitxor(x, bitshift(x, 17));
state = x;
u = double(bitshift(x, -11)) * (1.0 / 9007199254740992.0);
end
