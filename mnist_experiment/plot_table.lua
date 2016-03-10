require 'gnuplot'

local args = lapp[[
    -i,--input (string)    The data you want to plot
    -n,--name (string)     File name to save the image
]]

gnuplot.pngfigure(args.name)

a = torch.load(args.input)
print(args.input)
gnuplot.plot({
   torch.Tensor(a), '-'
})

gnuplot.plotflush()
