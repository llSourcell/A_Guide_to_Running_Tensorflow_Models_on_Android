package mariannelinhares.mnistandroid.views;
/*
   Copyright 2016 Narrative Nights Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   https://raw.githubusercontent.com/miyosuda/TensorFlowAndroidMNIST/master/app/src/main/java/jp/narr/tensorflowmnist/DrawRenderer.java
*/

//The Canvas class holds the "draw" calls. To draw something, you need 4 basic components: A Bitmap to hold the pixels,
// a Canvas to host the draw calls (writing into the bitmap), a drawing primitive (e.g. Rect, Path, text, Bitmap),
// and a paint (to describe the colors and styles for the drawing).
import android.graphics.Canvas;
//The Color class provides methods for creating, converting and manipulating colors.
import android.graphics.Color;
//The Paint class holds the style and color information about how to draw geometries, text and bitmaps.
import android.graphics.Paint;

import mariannelinhares.mnistandroid.views.DrawModel;

/**
 * Changed by marianne-linhares on 21/04/17.
 * https://raw.githubusercontent.com/miyosuda/TensorFlowAndroidMNIST/master/app/src/main/java/jp/narr/tensorflowmnist/DrawRenderer.java
 */

public class DrawRenderer {
    /**
     * Draw lines to canvas
     */
    //straight up drawing function to make the drawing visible to
    //the user. directly manipulates XML
    //given a canvas, a model drawing stored in memory, a color metadata,
    //start drawing!
    public static void renderModel(Canvas canvas, DrawModel model, Paint paint,
                                   int startLineIndex) {
        //minimize distortion artifacts
        paint.setAntiAlias(true);

        //get the size of the line to draw
        int lineSize = model.getLineSize();
        //given that size
        for (int i = startLineIndex; i < lineSize; ++i) {
            //get the whole line from the model object
            DrawModel.Line line = model.getLine(i);
            //set its color
            paint.setColor(Color.BLACK);
            //get the first of many lines that make up the overall line
            int elemSize = line.getElemSize();
            //if its empty, skip
            if (elemSize < 1) {
                continue;
            }
            // store that first line element in elem
            DrawModel.LineElem elem = line.getElem(0);
            //get its coordinates
            float lastX = elem.x;
            float lastY = elem.y;

            //for each coordinate in the line
            for (int j = 0; j < elemSize; ++j) {
                //get the next coordinate
                elem = line.getElem(j);
                float x = elem.x;
                float y = elem.y;
                //and draw the line between those two paints
                canvas.drawLine(lastX, lastY, x, y, paint);
                //store the coordinate as last and repeat
                //until the line is drawn
                lastX = x;
                lastY = y;
            }
        }
    }
}