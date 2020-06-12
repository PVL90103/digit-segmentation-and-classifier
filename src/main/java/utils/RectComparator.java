package utils;

import java.util.Comparator;
import org.bytedeco.opencv.opencv_core.Rect;

public class RectComparator implements Comparator<Rect> {

    @Override
    public int compare(Rect t1, Rect t2) {
        return Integer.valueOf(t1.x()).compareTo(t2.x());
    }
}