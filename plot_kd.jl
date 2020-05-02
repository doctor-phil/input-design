using Plots
f(x) = 1. .-x.^2
plot(f,-1,1,color=:black,linestyle=:dash,label="",minorgrid=false,grid=false,legendfont=(18,"times"))
x = [ -1; 0; 0; 0; 0; 1]
y = [ 0;0;NaN;1;0;0]
plot!(x,y,color=:black,label="")
x2 = [ 0 ]
y2 = [ 0 ]
plot!(x2,y2,marker=true,color=:white,markersize=4,label="")
x3 = [0]
y3 = [1]
plot!(x3,y3,marker=true,color=:black,markersize=4,label="")

title="diff.pdf"

savefig(title)
