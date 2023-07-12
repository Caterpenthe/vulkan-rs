#version 450

layout (location = 0) in vec4 inColor;

layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform UBO
{
//    vec3	iChannelResolution[4];	//image/buffer/sound	Input texture resolution for each channel
//    float	iChannelTime[4];	//image/buffer	Time for channel (if video or sound), in seconds
    vec4	iMouse;	//image/buffer	xy = current pixel coords (if LMB is down). zw = click pixel
    vec4	iDate;	//image/buffer/sound	Year, month, day, time in seconds in .xyzw
    vec3	iResolution;	//image/buffer	The viewport resolution (z is pixel aspect ratio, usually 1.0)
    float	iTime;	//image/sound/buffer	Current time in seconds
    float	iTimeDelta;	//image/buffer	Time it takes to render a frame, in seconds
    int	    iFrame;	//image/buffer	Current frame
    float	iFrameRate;	//image/buffer	Number of frames rendered per second
    float	iSampleRate;	//image/buffer/sound	The sound sample rate (typically 44100)


//    //sampler2D	iChannel{i}	//image/buffer/sound	Sampler for input textures i
} ubo;


// Created by inigo quilez - iq/2013
//   https://www.youtube.com/c/InigoQuilez
//   https://iquilezles.org/
// I share this piece (art and code) here in Shadertoy and through its Public API, only for educational purposes.
// You cannot use, sell, share or host this piece or modifications of it as part of your own commercial or non-commercial product, website or project.
// You can share a link to it or an unmodified screenshot of it provided you attribute "by Inigo Quilez, @iquilezles and iquilezles.org".
// If you are a teacher, lecturer, educator or similar and these conditions are too restrictive for your needs, please contact me and we'll work it out.

void main()
{
    vec2 fragCoord = ubo.iResolution.xy - gl_FragCoord.xy;
    vec2 p = (2.0*fragCoord-ubo.iResolution.xy)/min(ubo.iResolution.y,ubo.iResolution.x);

    // background color
    vec3 bcol = vec3(1.0,0.8,0.7-0.07*p.y)*(1.0-0.25*length(p));

    // animate
    float tt = mod(ubo.iTime,1.5)/1.5;
    float ss = pow(tt,.2)*0.5 + 0.5;
    ss = 1.0 + ss*0.5*sin(tt*6.2831*3.0 + p.y*0.5)*exp(-tt*4.0);
    p *= vec2(0.5,1.5) + ss*vec2(0.5,-0.5);

    // shape
    p.y -= 0.25;
    float a = atan(p.x,p.y)/3.141593;
    float r = length(p);
    float h = abs(a);
    float d = (13.0*h - 22.0*h*h + 10.0*h*h*h)/(6.0-5.0*h);

    // color
    float s = 0.75 + 0.75*p.x;
    s *= 1.0-0.4*r;
    s = 0.3 + 0.7*s;
    s *= 0.5+0.5*pow( 1.0-clamp(r/d, 0.0, 1.0 ), 0.1 );
    vec3 hcol = vec3(1.0,0.4*r,0.3)*s;

    vec3 col = mix( bcol, hcol, smoothstep( -0.01, 0.01, d-r) );

    outFragColor = vec4(col,1.0);
}