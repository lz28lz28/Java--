//import java.util.*;
//public class Main {
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int N = sc.nextInt();
//        int[][] list = new int[N][2];
//        for (int i = 0; i < N; i++) {
//            list[i][0] = sc.nextInt();
//            list[i][1] = sc.nextInt();
//        }
//        for (int i = 0; i < N; i++) {
//            for (int j = i+1; j < N; j++) {
//                if(list[i][0]>=list[j][0]&&list[i][1]<=list[j][1]) System.out.println(true);
//            }
//        }
//        System.out.println(false);
//    }
//}

////两个线程打印奇偶或者ab
//public class Main {
//
//    static boolean flag = false;
//    static int count =0;
//
//    public static void main(String[] args) {
//
//        Object lock = new Object();
//
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i=0;i<1000;i++) {
//                    synchronized (lock){
//                        if(flag){
//                            try {
//                                lock.wait();
//                            } catch (InterruptedException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                        flag=true;
//                        if(count<100){
//                            System.out.println(Thread.currentThread().getName()+":"+count);
//                            count++;
//                        }
//                        //System.out.println("a");
//                        lock.notify();
//                    }
//                }
//            }
//        }).start();
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i=0;i<1000;i++) {
//                    synchronized (lock){
//                        if(!flag){
//                            try {
//                                lock.wait();
//                            } catch (InterruptedException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                        flag=false;
//                        if(count<100){
//                            System.out.println(Thread.currentThread().getName()+":"+count);
//                            count++;
//                        }
//                        //System.out.println("b");
//                        lock.notify();
//                    }
//                }
//            }
//        }).start();
//    }
//}


////三个线程打印abc，synchronized法，注意，不能用if了，要改为while
//public class Main {
//
//    static boolean flag = false;
//    static boolean flag2 = false;//也可以用num来具体指定哪个线程运行
//
//    public static void main(String[] args) {
//
//        Object lock = new Object();
//
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i=0;i<3;i++) {
//                    synchronized (lock){
//                        while(!(flag==false&&flag2==false)){ //此处if要改为while，后面同理
//                            try {
//                                lock.wait();
//                            } catch (InterruptedException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                        flag=true;
//                        flag2=false;
//                        System.out.println("a");
//                        lock.notifyAll();
//                    }
//                }
//            }
//        }).start();
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i=0;i<3;i++) {
//                    synchronized (lock){
//                        while(!(flag==true&&flag2==false)){
//                            try {
//                                lock.wait();
//                            } catch (InterruptedException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                        flag=true;
//                        flag2=true;
//                        System.out.println("b");
//                        lock.notifyAll();
//                    }
//                }
//            }
//        }).start();
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i=0;i<3;i++) {
//                    synchronized (lock){
//                        while(!(flag==true&&flag2==true)){
//                            try {
//                                lock.wait();
//                            } catch (InterruptedException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                        flag=false;
//                        flag2=false;
//                        System.out.println("c");
//                        lock.notifyAll();
//                    }
//                }
//            }
//        }).start();
//    }
//}



////三个线程打印abc，lock法
//import java.util.concurrent.locks.Condition;
//import java.util.concurrent.locks.ReentrantLock;
//
//public class Main {
//
//    private static ReentrantLock lock = new ReentrantLock();
//    private static Condition condition1 = lock.newCondition();
//    private static Condition condition2 = lock.newCondition();
//    private static Condition condition3 = lock.newCondition();
//    private static int count = 0;
//
//    public static void main(String[] args) throws InterruptedException {
//
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i = 0;i < 10;i++) {
//                    lock.lock();
//                    try {
//                        if (count % 3 != 0) {
//                            condition1.await();
//                        }
//                        System.out.print("A");
//                        count++;
//                        condition2.signal();
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    } finally {
//                        lock.unlock();
//                    }
//                }
//            }
//        }).start();
//
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i = 0;i < 10;i++) {
//                    lock.lock();
//                    try {
//                        if (count % 3 != 1) {
//                            condition2.await();
//                        }
//                        System.out.print("B");
//                        count++;
//                        condition3.signal();
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    } finally {
//                        lock.unlock();
//                    }
//                }
//            }
//        }).start();
//
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i = 0;i < 10;i++) {
//                    lock.lock();
//                    try {
//                        if (count % 3 != 2) {
//                            condition3.await();
//                        }
//                        System.out.print("C");
//                        System.out.println("============");
//                        count++;
//                        condition1.signal();
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    } finally {
//                        lock.unlock();
//                    }
//                }
//            }
//        }).start();
//    }
//}


////lambda表达式
//// 1.1使用匿名内部类
//new Thread(new Runnable() {
//@Override
//public void run() {
//        System.out.println("Hello world !");
//        }
//        }).start();
//
//// 1.2使用 lambda expression
//        new Thread(() -> System.out.println("Hello world !")).start();
//
//// 2.1使用匿名内部类
//        Runnable race1 = new Runnable() {
//@Override
//public void run() {
//        System.out.println("Hello world !");
//        }
//        };
//
//// 2.2使用 lambda expression
//        Runnable race2 = () -> System.out.println("Hello world !");
//
//// 直接调用 run 方法(没开新线程哦!)
//        race1.run();
//        race2.run();


//比较方法
//import java.util.*;
//class MyComparator implements Comparator<Integer>{
//
//    @Override
//    public int compare(Integer o1, Integer o2) {
//        return o2-o1;
//    }
//}
//public class Main {
//
//    public static void main(String[] args) {
//        Integer[] a = {4,6,8,2,1,4,9,2,4};
//        //Arrays.sort(a,Collections.reverseOrder());
//        Arrays.sort(a,new Comparator<Integer>(){
//            @Override
//            public int compare(Integer o1, Integer o2) {
//                return o2-o1;
//            }
//        });
//        for(int arr:a) {
//            System.out.print(arr + " ");
//        }
//    }
//}

//public class Main{
//    public static void main(String[] args){
//        String a = "123";
//        int b = Integer.parseInt(a);
//        System.out.println(b);
//    }
//}

////锁lock，线程池
//import java.util.concurrent.ExecutorService;
//import java.util.concurrent.locks.ReentrantLock;
//import java.util.concurrent.Executors;
//
//class LockExample {
//
//    private ReentrantLock lock = new ReentrantLock();
//
//    public void func() {
//        lock.lock();
//        try {
//            for (int i = 0; i < 100; i++) {
//                System.out.print(Thread.currentThread().getName()+":"+i + " ");
//            }
//        } finally {
//            lock.unlock(); // 确保释放锁，从而避免发生死锁。
//        }
//    }
//}
//
//public class Main{
//    public static void main(String[] args){
//        LockExample lockExample = new LockExample();
//        ExecutorService executorService = Executors.newFixedThreadPool(5);
//        executorService.execute(() -> lockExample.func());
//        executorService.execute(() -> lockExample.func());
//        executorService.execute(new Runnable() {
//            @Override
//            public void run() {
//                lockExample.func();
//            }
//        });
//        System.out.println("##############");
//        executorService.shutdown();
//    }
//}

////手动创建线程池
//import java.util.concurrent.*;
//class SubThread implements Runnable {
//    @Override
//    public void run() {
//        try {
//            System.out.println(Thread.currentThread().getName());
//            Thread.sleep(100);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//    }
//}
//
//public class Main{
//    public static int corePoolSize = 10;
//    public static int maximumPoolSizeSize = 20;
//    public static long keepAliveTime = 1;
//    public static BlockingQueue<Runnable> workQueue = new ArrayBlockingQueue<>(5);
//    public static void main(String[] args) {
//        ThreadPoolExecutor mythreadpool = new ThreadPoolExecutor(corePoolSize,maximumPoolSizeSize,keepAliveTime,TimeUnit.SECONDS,workQueue, Executors.defaultThreadFactory(),new ThreadPoolExecutor.DiscardPolicy());
//        for (int i = 0; i < 26; i++) {
//            mythreadpool.execute(new SubThread());
//        }
//        mythreadpool.shutdown();
//    }
//}


////正则表达式
//import java.util.regex.Matcher;
//import java.util.regex.Pattern;
//public class Main {
//    public static void main(String[] args) {
//        String s = "abCDef123";
//        Pattern pattern = Pattern.compile( "[a-z]{2}(?=\\d+)");
//        Matcher matcher = pattern.matcher(s);
//        if(matcher.find())
//            System.out.println(matcher.group());
//    }
////    public static void main(String[] args){
////        String content = "I am noob " + "from runoob.com.";
////        String pattern = ".*runoob.*";
////        boolean isMatch = Pattern.matches(pattern, content);
////        System.out.println("字符串中是否包含了 'runoob' 子字符串? " + isMatch);
////    }
//}


////threadlocal的学习和使用
//public class Main{
//    static class myThreadlocal extends ThreadLocal{
//        @Override
//        protected Object initialValue() {
//            return "初始值";
//        }
//    }
//    public static void main(String[] args) {
//        myThreadlocal myThreadlocal1 = new myThreadlocal();
//        myThreadlocal myThreadlocal2 = new myThreadlocal();
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
////                myThreadlocal1.set("thread01");
////                myThreadlocal1.remove();
////                myThreadlocal2.set("thread02");
//                for (int i = 0; i < 5; i++) {
//                    System.out.println(Thread.currentThread().getName()+":"+ myThreadlocal1.get()+","+ myThreadlocal2.get());
//                }
//            }
//        }).start();
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                myThreadlocal1.set("thread11");
//                myThreadlocal2.set("thread12");
//                for (int i = 0; i < 5; i++) {
//                    System.out.println(Thread.currentThread().getName()+":"+ myThreadlocal1.get()+","+ myThreadlocal2.get());
//                }
//            }
//        }).start();
//    }
//}

////深浅拷贝实例
//class teacher implements Cloneable{
//    public teacher(String name){
//        this.name=name;
//    }
//    private String  name;
//
//    public String getName() {
//        return name;
//    }
//
//    public void setName(String name) {
//        this.name = name;
//    }
//
//    @Override
//    protected Object clone() throws CloneNotSupportedException {
//        return super.clone();
//    }
//}
//
//class student implements Cloneable{
//    public student(int id, String name, teacher myteacher){
//        this.id=id;
//        this.name=name;
//        this.myteacher=myteacher;
//    }
//
//    private int id;
//    private String name;
//    private teacher myteacher;
//
//    public int getId() {
//        return id;
//    }
//
//    public void setId(int id) {
//        this.id = id;
//    }
//
//    public String getName() {
//        return name;
//    }
//
//    public void setName(String name) {
//        this.name = name;
//    }
//
//    public teacher getMyteacher() {
//        return myteacher;
//    }
//
//    public void setMyteacher(teacher myteacher) {
//        this.myteacher = myteacher;
//    }
////浅拷贝写法
////    @Override
////    protected Object clone() throws CloneNotSupportedException {
////        return super.clone();
////    }
//
//    @Override
//    protected Object clone() throws CloneNotSupportedException {
//        student student = (student) super.clone();
//        student.myteacher = (teacher) myteacher.clone();
//        return student;
//    }
//}
//
//public class Main{
//    public static void main(String[] args) {
//        teacher teacher = new teacher("李猛");
//        student student = new student(8,"小明",teacher);
//        try {
//            student studentclone = (student) student.clone();
//            studentclone.setId(9);
//            studentclone.setName("克隆版小明");
//            studentclone.getMyteacher().setName("克隆人改名字");
//            System.out.println(student);
//            System.out.println(studentclone);
//        } catch (CloneNotSupportedException e) {
//            e.printStackTrace();
//        }
//    }
//}


//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.Queue;
//
//class myclassloader extends ClassLoader{
//    private String name;
//
//    @Override
//    protected Class<?> findClass(String name) throws ClassNotFoundException {
//        return super.findClass(name);
//    }
//
//    @Override
//    public Class<?> loadClass(String name) throws ClassNotFoundException {
//        return super.loadClass(name);
//    }
//}
//
//public class Main{
//    public static void main(String[] args) {
//        HashMap<Integer,Integer> map = new HashMap<>();
//        ArrayList<Integer> testlist = new ArrayList<>();
//
//    }
//}

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;
import java.util.concurrent.ThreadPoolExecutor;

public class Main{
    public static void main(String[] args) {
        String version1 = "0.1";
        String[] s1 = version1.split("\\.");
        System.out.println(s1.length);
    }
}
