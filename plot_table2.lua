require 'gnuplot'
require 'paths'

local args = lapp[[
    -i,--input1 (string)    The data 1 you want to plot
    -j,--input2 (default '')    The data 2 you want to plot
    -p,--powerRecord (default 'nil') The record of power iteration call
    -n,--name (string)     File name to save the image
    -s,--save (default 'img') Folder to save
    -e,--epochPlot plot for test/train errors per epoch
    -t,--epochPlotTensor  plot for test/train error/acc that is stored as lua table
    -x,--xlabel (string)  xlabel
    -y,--ylabel (string)  ylabel
]]


--if(args.help) then print(args); os.exit() end


local filename  =  paths.concat(args.save, args.name)

print(args.epochPlotTensor)

if args.epochPlot then
    gnuplot.pngfigure(filename)
    -- Creates a figure directly on the png file given with args.name. 
    -- This uses Gnuplot terminal png, or pngcairo if available.

    require 'csvigo' 
    print("ok")
    local test = csvigo.load(args.input1)
    test = torch.Tensor(test["test_acc"])
    local train = csvigo.load(args.input2)
    train = torch.Tensor(train["train_acc"])

    --power = torch.load(args.powerRecord)


    gnuplot.plot({test, "-"},{train, "-"})

    gnuplot.plotflush()  

elseif args.epochPlotTensor then
    print(filename)
    gnuplot.epsfigure(filename)
    print(args.input1)
    local train = torch.Tensor(torch.load(args.input1))
    print(args.input2)
    local test = torch.Tensor(torch.load(args.input2))

    gnuplot.xlabel(args.xlabel)
    gnuplot.ylabel(args.ylabel)

    gnuplot.plot({"test",test, "-"},{"train", train, "-"})

    gnuplot.plotflush()
else
    gnuplot.pngfigure(filename)
    a = torch.load(args.input1)

    gnuplot.xlabel(args.xlabel)
    gnuplot.ylabel(args.ylabel)

    gnuplot.plot({torch.Tensor(a), '+'})

    gnuplot.plotflush()
end

