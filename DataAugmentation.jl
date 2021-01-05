"""
Data Augmentation module includes
- random cropping
- random color distortions
- random Gaussian blur 
"""
#using Pkg; for p in ["Images", "TestImages", "Distributions"]; Pkg.add(p); end
using Images, TestImages, Statistics, Distributions

"""
Randomly apply function func to x with probability p.
"""
function random_apply(func, p, x)
    return rand() .< p ? func(x) : x
end

"""
Transform RGB image to grayscale image
If keep_channels=true, channel number will be tiled to 3.
"""
function to_grayscale(image; keep_channels=true)
    image = Gray.(image)
    if keep_channels
        image = repeat(image, outer = [1, 1, 3])
    end
    return image
end


"""
Random contrast within given lower and upper bound 
"""
function random_contrast(image, lower, upper)
    r_cont = rand(lower:0.00001:upper)
    #r_cont = rand(Uniform(lower,upper))
    mean_f =  mean(image, dims=(1,2))
    f_image = (image .- mean_f)*r_cont .+ mean_f
    return f_image
end


"""
Random brightness within given max_delta
"""
function random_brightness(image, max_delta)
    if max_delta < 0
        raise("max_delta must be non-negative.")
    else
        r_bright = rand(-max_delta:0.00001:max_delta)
        #b_image =  permutedims(channelview(image), (2, 3, 1)) .+ r_bright
        b_image =  image .+ r_bright
    end
    return b_image
end


