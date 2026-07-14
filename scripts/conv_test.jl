using FFTW, Novikov, Plots

f(x) = cos(2x)
g(x) = 4sin(4x)

L = 4pi
N_f = 1024
N = 2N_f + 1

h(x) = f(x) * g(x)

x, f_d = dscrt(f, L, N)
_, g_d = dscrt(g, L, N)
_, h_d = dscrt(h, L, N)

fhat = rfft(f_d)
ghat = rfft(g_d)

h_hat = zeros(ComplexF64, N_f + 1)

function get_hat(arr, i)
    if i == 0
        return 0
    elseif i < 0
        return conj(arr[-i+1])
    else
        return arr[i+1]
    end
end
for k ∈ 0:N_f
    for l ∈ max(-N_f, -N_f+k):min(N_f, N_f+k)
        h_hat[k+1] += get_hat(fhat, k-l) * get_hat(ghat, l)
    end
end

h_hat ./= N
h_f = irfft(h_hat, N)

display(plot(x, f_d, label="f(x)"))
display(plot(x, g_d, label="g(x)"))
display(plot(x, h_d, label="h(x)"))

display(plot(x, (h_d .- h_f), label="h-fourier"))




