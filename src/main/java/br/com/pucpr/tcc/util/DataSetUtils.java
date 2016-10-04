package br.com.pucpr.tcc.util;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by douglas on 9/29/16.
 */
public class DataSetUtils {

    private static final Pattern pattern;
    private static List<String> lines = new ArrayList<>();

    static {
        pattern = Pattern.compile("((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)",
                Pattern.CASE_INSENSITIVE);
    }

    public static void cleanCorpusTwitter(String inputPath, String outputPath, Integer workers) {
        readLines(openFile(inputPath));
        BufferedWriter output = createOutputFile(outputPath);

        Double slice = Double.valueOf(lines.size() / workers);
        if (slice - slice.intValue() > 0D) {
            workers += 1;
        }

        ExecutorService executorService = Executors.newFixedThreadPool(workers);
        List<Future<List<String>>> threads = new ArrayList<>();
        for (int i = 0; i < workers; i++) {
            threads.add(executorService.submit(getThread(i, slice.intValue())));
        }
        List<String> newLines = new ArrayList<>();

        try {
            for (int i = 0; i < workers; i++) {
                Future<List<String>> future = threads.get(i);
                while (true) {
                    if (future.isDone()) break;
                    Thread.currentThread().sleep(2000);
                }
                newLines.addAll(future.get());
            }
            for (String str : newLines) {
                output.write(str);
                output.newLine();
            }
            output.close();
            executorService.shutdown();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }

        Future<List<String>> future = threads.get(0);
    }

    private static void readLines(BufferedReader bufferedReader) {
        bufferedReader.lines().forEach(str -> lines.add(str));
    }

    private static Callable<List<String>> getThread(int start, int blockSize) {
        List<String> slice = sliceLines(start, blockSize);
        Callable<List<String>> run = () -> {
            List<String> newLines = new ArrayList<>();
            slice.stream().forEach(line -> {
                String[] tokens = line.split(",");

                Matcher match = pattern.matcher(tokens[5]);
                for (int i = 0; match.find(); i++) {
                    tokens[5] = tokens[5].replace(match.group(i), "");
                }

                tokens[5] = tokens[5].toLowerCase();
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < tokens[5].length(); i++) {
                    int ascii = (int) tokens[5].charAt(i);
                    if (ascii >= 97 && ascii <= 122) {
                        stringBuilder.append(tokens[5].charAt(i));
                    } else if (ascii == 32) {
                        stringBuilder.append(tokens[5].charAt(i));
                    } else if (ascii == 35) {
                        stringBuilder.append(tokens[5].charAt(i));
                    } else if (ascii == 64) {
                        stringBuilder.append(tokens[5].charAt(i));
                    }
                }

                //String.format("%s", line);
                newLines.add(stringBuilder.toString());
            });
            return newLines;
        };
        return run;
    }

    private static List<String> sliceLines(int num, int blockSize) {
        int start = num * blockSize;
        int end = (start + blockSize) > lines.size() ? lines.size() - 1 : (start + blockSize);
        return lines.subList(start, end);
    }

    private static BufferedReader openFile(String inputPath) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(inputPath));
            return bufferedReader;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private static BufferedWriter createOutputFile(String outputPath) {
        try {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputPath));
            return bufferedWriter;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
