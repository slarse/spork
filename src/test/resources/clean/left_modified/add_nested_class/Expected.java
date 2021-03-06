public class Adder {
    public int add(int a, int b) {
        InputChecker.checkInput(a, b);
        return a + b;
    }

    private static class InputChecker {
        static int checkInput(int a, int b) {
            if (b <= a) {
                throw new IllegalArgumentException("a must be smaller than b");
            }
        }
    }
}
