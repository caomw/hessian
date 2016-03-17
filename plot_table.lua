require 'gnuplot'
require 'paths'

local args = lapp[[
    -i,--input1 (string)    The data 1 you want to plot
    -j,--input2 (default 'nil')    The data 2 you want to plot
    -n,--name (string)     File name to save the image
    -s,--save (default 'img') Folder to save
    -h,--help (default true) show help document
    -e,--epochPlot (default false) plot for test/train errors per epoch
]]


--if(args.help) then print(args); os.exit() end


local filename  =  paths.concat(args.save, args.name)

gnuplot.pngfigure(filename)
-- Creates a figure directly on the png file given with args.name. 
-- This uses Gnuplot terminal png, or pngcairo if available.


if(args.epochPlot == "true") then
    require 'csvigo' 
    print("ok")
    local test = csvigo.load(args.input1)
    test = torch.Tensor(test["test_acc"])
    local train = csvigo.load(args.input2)
    train = torch.Tensor(train["train_acc"])

    gnuplot.plot({test, "-"},{train, "-"})

    gnuplot.plotflush()  

else
    a = torch.load(args.input1)

    gnuplot.plot({
        torch.Tensor(a), '+'
    })

    gnuplot.plotflush()
end

