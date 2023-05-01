from math import sin, cos, pi
from .style import _color_palette

def _pie_chart_svg(cx, cy, r, percentages:dict):
    '''Returns SVG code to draw a pie chart using a rainbow pallette'''
    ret = ''
    # Sorting the items to get a deterministic view of the dict
    percentages_sorted = sorted(percentages.items())
    # Retrieving the maximum to mark the chart with the majority vote
    max_item = max(enumerate(percentages_sorted), key=lambda item: item[1][1])

    # Drawing the outer circle indicating the majority vote
    ret += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_color_palette(len(percentages))[max_item[0]]}"/>'
    # Drawing the seperating white circle
    ret += f'<circle cx="{cx}" cy="{cy}" r="{r*0.9}" fill="hsl({0},0%,100%)"/>'

    # If we have no entropy, we just draw a complete circle
    if max_item[1][1] == 1:
        ret += f'<circle cx="{cx}" cy="{cy}" r="{r*0.85}" fill="{_color_palette(len(percentages))[max_item[0]]}"/>'
        # Draw white circle in the middle to allow for text
        ret += f'<circle cx="{cx}" cy="{cy}" r="{r*0.6}" fill="hsl({0},0%,100%)"/>'
        return ret
    
    # Draw Actual pie chard based on percentages
    total = -.25
    for i, percentage in enumerate(percentages_sorted):
        # Very long string. Long explanation. We draw a path from the center, over an arc, back to the center.
        # M moves the cursor to the center
        # L draws a line to the start of the label's segment
        # A draws an art from the start to the end of the segment
        # - The "if" is needed, because in the case of >0.5 percentage, we have to choose the longer arc
        # z moves the cursor back to the center and closes the arc
        ret += f'<path d="M{cx},{cy} L{round(cos(total*2*pi)*r*0.85+cx,10)},{round(sin(total*2*pi)*r*0.85+cy,10)} A{r*0.85},{r*0.85} 0 {0 if percentage[1] <= 0.5 else 1},1 {round(cos((total+percentage[1])*2*pi)*r*0.85+cx,10)},{round(sin((total+percentage[1])*2*pi)*r*0.85+cy,10)} z" fill="{_color_palette(len(percentages))[i]}" stroke="{_color_palette(len(percentages))[i]}" stroke-width="0.1"/>'
        total += percentage[1]
    
    # Draw white circle in the middle to allow for text
    ret += f'<circle cx="{cx}" cy="{cy}" r="{r*0.6}" fill="hsl({0},0%,100%)"/>'
    return ret

def _legend_svg(labels, target=None):
        '''Returns an SVG string of the legend, along with it's width and height'''
        labels = sorted(labels)
        y_offset = 0
        height = 20*len(labels) 
        if target is not None:
            y_offset = 25
            height += 30
        width = 150 
        svgstr = f"""<svg width="{width+2}" height="{height+2}" xmlns="http://www.w3.org/2000/svg">
        <rect rx="5" ry="5" width="{width}" height="{height}" x="1" y="1" stroke="#AAAAAA" stroke-width="2" fill="white"/>"""
        if target is not None:
            svgstr +=f'<text x="75" y="{15}" font-family="Arial, Helvetica, sans-serif" font-weight="bold" dominant-baseline="central" text-anchor="middle">{target}</text>'
        for i, (color, label) in enumerate(zip(_color_palette(len(labels)), labels)):
            svgstr += f'<line stroke="{color}" stroke-width="2.5" x1="10" x2="30" y1="{10+i*20+y_offset}" y2="{10+i*20+y_offset}"/>'
            svgstr +=f'<text x="35" y="{10+i*20 + y_offset}" font-family="Arial, Helvetica, sans-serif" dominant-baseline="central">{label}</text>'

        svgstr += "</svg>"
        return svgstr, width+2, height+2