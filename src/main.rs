use leet_code::structures::{ListNode, TreeNode};
use std::cell::RefCell;
use std::collections::HashMap;
use std::option::Option::Some;
use std::rc::Rc;

fn main() {
    println!("Hello, world!");
    // Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    // Output: 7 -> 0 -> 8
    let mut l1 = Some(Box::new(ListNode::new(2)));
    l1.as_mut().unwrap().next = Some(Box::new(ListNode::new(4)));
    l1.as_mut().unwrap().next.as_mut().unwrap().next = Some(Box::new(ListNode::new(3)));
    let mut l2 = Some(Box::new(ListNode::new(5)));
    l2.as_mut().unwrap().next = Some(Box::new(ListNode::new(6)));
    l2.as_mut().unwrap().next.as_mut().unwrap().next = Some(Box::new(ListNode::new(4)));
    let l3 = Solution::add_two_numbers(l1, l2);
    println!("{:?}", l3);
    let s = String::from("abbbcccdefg");
    let len = Solution::length_of_longest_substring(s);
    println!("{}", len);
    let v1 = vec![1, 3];
    let v2 = vec![2];
    let res = Solution::find_median_sorted_arrays(v1, v2);
    println!("{}, ======{}", res, 5 / 2 as i32)
}

struct Solution;
#[allow(dead_code)]
#[allow(unused_assignments)]
impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut map: HashMap<i32, usize> = HashMap::with_capacity(nums.len());
        for (i, &v) in nums.iter().enumerate() {
            let anther = target - v;
            if let Some(&j) = map.get(&anther) {
                return vec![j as i32, i as i32];
            }
            map.insert(v, i);
        }
        vec![]
    }

    pub fn add_two_numbers(
        l1: Option<Box<ListNode>>,
        l2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        let (mut l1, mut l2) = (l1, l2);
        if let (Some(_), Some(_)) = (&l1, &l2) {
            let mut dummy = Some(Box::new(ListNode::new(0)));
            let mut current = &mut dummy;
            let mut carry = 0;
            loop {
                if let (None, None) = (&l1, &l2) {
                    break;
                } else {
                    let (mut x, mut y) = (0, 0);
                    if let Some(n1) = &l1 {
                        x = n1.val;
                    }
                    if let Some(n2) = &l2 {
                        y = n2.val;
                    }
                    current.as_mut().unwrap().next =
                        Some(Box::new(ListNode::new((x + y + carry) % 10)));
                    current = &mut current.as_mut().unwrap().next;
                    carry = (x + y + carry) / 10;
                    if let Some(_) = &l1 {
                        l1 = l1.unwrap().next;
                    }
                    if let Some(_) = &l2 {
                        l2 = l2.unwrap().next;
                    }
                }
            }
            // 处理最终进位
            if carry > 0 {
                current.as_mut().unwrap().next = Some(Box::new(ListNode::new(carry % 10)));
            }
            dummy.unwrap().next
        } else {
            return None;
        }
    }
    pub fn length_of_longest_substring(s: String) -> i32 {
        use std::cmp::max;
        if s.len() == 0 {
            return 0;
        }
        let mut freq = [0; 128];
        let (mut res, mut left, mut right) = (0, 0, 0);
        while right < s.len() {
            if freq[*s.as_bytes().get(right).unwrap() as usize] == 0 {
                freq[*s.as_bytes().get(right).unwrap() as usize] += 1;
                right += 1;
            } else {
                freq[*s.as_bytes().get(left).unwrap() as usize] -= 1;
                left += 1;
            }
            res = max(res, right - left);
        }
        res as i32
    }
    //O(log(min(m,n)))
    pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        // 假设nums1的长度较短
        use std::cmp::{max, min};
        let (nums1, nums2) = (nums1, nums2);
        let (len1, len2) = (nums1.len(), nums2.len());
        if len1 > len2 {
            return Self::find_median_sorted_arrays(nums2, nums1);
        }

        // low, high, k, nums1Mid, nums2Mid := 0, len(nums1), (len(nums1)+len(nums2)+1)>>1, 0, 0
        let (mut low, mut high, k, mut nums1_mid, mut nums2_mid) =
            (0, len1, (len1 + len2 + 1) >> 1, 0, 0);
        // for low <= high {
        while low <= high {
            //     // nums1:  ……………… nums1[nums1Mid-1] | nums1[nums1Mid] ……………………
            //     // nums2:  ……………… nums2[nums2Mid-1] | nums2[nums2Mid] ……………………
            // 需要注意的是rust中移位运算符的优先级与c语言(go)略有不同
            nums1_mid = low + ((high - low) >> 1);
            // k ：列表合并后的分界线位置
            nums2_mid = k - nums1_mid;
            if nums1_mid > 0 && nums1.get(nums1_mid - 1) > nums2.get(nums2_mid) {
                high = nums1_mid - 1;
            } else if nums1_mid != len1 && nums1.get(nums1_mid) < nums2.get(nums2_mid - 1) {
                low = nums1_mid + 1;
            } else {
                //     // 找到合适的划分了，需要输出最终结果了
                //     // 分为奇数偶数 2 种情况
                break;
            }
        }
        let (mut mid_left, mut mid_right) = (0, 0);
        if nums1_mid == 0 {
            mid_left = *nums2.get(nums2_mid - 1).unwrap();
        } else if nums2_mid == 0 {
            mid_left = *nums1.get(nums1_mid - 1).unwrap();
        } else {
            mid_left = max(
                *nums1.get(nums1_mid - 1).unwrap(),
                *nums2.get(nums2_mid - 1).unwrap(),
            );
        }

        if (len1 + len2) & 1 == 1 {
            return mid_left as f64;
        }

        if nums1_mid == len1 {
            mid_right = *nums2.get(nums2_mid).unwrap();
        } else if nums2_mid == len2 {
            mid_right = *nums1.get(nums1_mid).unwrap();
        } else {
            mid_right = min(
                *nums1.get(nums1_mid).unwrap(),
                *nums2.get(nums2_mid).unwrap(),
            );
        }
        ((mid_left + mid_right) as f64) / 2.0
    }
    // 组合问题：dfs(深度优先搜索) + 回溯
    pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
        if n <= 0 || k <= 0 || k > n {
            return vec![];
        }
        let (mut c, mut res) = (vec![], vec![]);
        Self::dfs(n, k, 1, &mut c, &mut res);
        res
    }
    fn dfs(n: i32, k: i32, start: i32, c: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if c.len() as i32 == k {
            let mut b = Vec::with_capacity(c.len());
            b = c.clone();
            return res.push(b);
        }
        for i in start..=(n - (k - c.len() as i32) + 1) {
            c.push(i);
            Self::dfs(n, k, i + 1, c, res);
            c.pop();
        }
    }

    pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut candidates = candidates;
        if candidates.len() == 0 {
            return vec![];
        }
        let (mut c, mut res) = (vec![], vec![]);
        candidates.sort();
        Self::dfs1(&candidates, target, 0, &mut c, &mut res);
        res
    }
    fn dfs1(nums: &Vec<i32>, target: i32, index: i32, c: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if target <= 0 {
            if target == 0 {
                let mut b = Vec::with_capacity(c.len());
                b = c.clone();
                res.push(b);
            }
            return;
        }
        for i in index..nums.len() as i32 {
            c.push(*nums.get(i as usize).unwrap());
            Self::dfs1(nums, target - nums.get(i as usize).unwrap(), i, c, res);
            c.pop();
        }
    }
    pub fn combination_sum2(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut candidates = candidates;
        if candidates.len() == 0 {
            return vec![];
        }
        let (mut c, mut res) = (vec![], vec![]);
        candidates.sort();
        Self::dfs2(&candidates, target, 0, &mut c, &mut res);
        res
    }
    fn dfs2(nums: &Vec<i32>, target: i32, index: i32, c: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if target == 0 {
            let mut b = Vec::with_capacity(c.len());
            b = c.clone();
            return res.push(b);
        }
        for i in index..nums.len() as i32 {
            // 这里是去重的关键逻辑,本次不取重复数字，下次循环可能会取重复数字
            if i > index && nums.get(i as usize) == nums.get((i - 1) as usize) {
                continue;
            }
            if target >= *nums.get(i as usize).unwrap() {
                c.push(*nums.get(i as usize).unwrap());
                Self::dfs2(nums, target - nums.get(i as usize).unwrap(), i + 1, c, res);
                c.pop();
            }
        }
    }
    pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
        if k == 0 {
            return vec![];
        }
        let (mut c, mut res) = (vec![], vec![]);
        Self::dfs3(k, n, 1, &mut c, &mut res);
        res
    }
    fn dfs3(k: i32, target: i32, index: i32, c: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if target == 0 {
            if c.len() == k as usize {
                let mut b = Vec::with_capacity(c.len());
                b = c.clone();
                res.push(b);
            }
            return;
        }
        for i in index..10 {
            if target >= i {
                c.push(i);
                Self::dfs3(k, target - i, i + 1, c, res);
                c.pop();
            }
        }
    }
    // 递归
    pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        // 左根右
        let mut res = vec![];
        Self::inorder(root, &mut res);
        res
    }
    fn inorder(root: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
        match root {
            None => {
                return;
            }
            Some(node) => {
                // 中序遍历就是在遍历左子树和遍历右子树的中间执行操作.
                // 前序遍历就是在遍历左子树和遍历右子树之前执行操作.
                // 后序遍历就是在遍历左子树和遍历右子树之后执行操作.
                Self::inorder(node.borrow_mut().left.take(), result);
                result.push(node.borrow_mut().val);
                Self::inorder(node.borrow_mut().right.take(), result);
            }
        }
    }
    // 循环
    pub fn inorder_traversal_for(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        // 左根右
        let mut res = vec![];
        let mut stack = vec![];
        let mut root = root;
        while root.is_some() || !stack.is_empty() {
            while let Some(node) = root {
                root = node.borrow_mut().left.take();
                stack.push(node);
            }
            root = stack.pop();
            res.push(root.as_ref().unwrap().borrow_mut().val);
            root = root.unwrap().borrow_mut().right.take();
            // is the same as above
            // if let Some(node) = stack.pop() {
            //     // node.borrow().val == (*node).borrow().val
            //     res.push(node.borrow().val);
            //     root = node.borrow_mut().right.take();
            // }
        }
        res
    }
    // 前中后序遍历   // 层序遍历
    pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        // 方法一：
        // if let Some(r) = root.as_ref() {
        //     let (left, right) = (r.borrow().left.clone(), r.borrow().right.clone());
        //     r.borrow_mut().right = Solution::invert_tree(left);
        //     r.borrow_mut().left = Solution::invert_tree(right);
        // }
        // root
        //  =============方法二
        if let Some(root) = root {
            let mut root_mut = root.borrow_mut();
            let temp = root_mut.left.clone();
            root_mut.left = root_mut.right.clone();
            root_mut.right = temp;

            Self::invert_tree(root_mut.left.clone());
            Self::invert_tree(root_mut.right.clone());
            drop(root_mut);
            return Some(root);
        }
        root
    }
    pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
        let mut row = vec![vec![false; 9]; 9];
        let mut col = vec![vec![false; 9]; 9];
        let mut block = vec![vec![false; 9]; 9];
        let mut rest = vec![];
        for i in 0..9 {
            for j in 0..9 {
                match board[i][j] {
                    '.' => rest.push((i, j)),
                    _ => {
                        let n = (board[i][j] as u8 - b'1') as usize;
                        row[i][n] = true;
                        col[j][n] = true;
                        block[i / 3 * 3 + j / 3][n] = true;
                    }
                }
            }
        }
        Self::dfs4(board, &rest, &mut row, &mut col, &mut block);
    }
    fn dfs4(
        board: &mut Vec<Vec<char>>,
        rest: &[(usize, usize)],
        row: &mut Vec<Vec<bool>>,
        col: &mut Vec<Vec<bool>>,
        block: &mut Vec<Vec<bool>>,
    ) -> bool {
        if let Some(&(i, j)) = rest.first() {
            for x in 0..9 {
                if !row[i][x] && !col[j][x] && !block[i / 3 * 3 + j / 3][x] {
                    // 剪枝
                    // 做选择
                    row[i][x] = true;
                    col[j][x] = true;
                    block[i / 3 * 3 + j / 3][x] = true;
                    board[i][j] = (x as u8 + b'1') as char;
                    // 回溯
                    if Self::dfs4(board, &rest[1..], row, col, block) {
                        return true;
                    }
                    // 撤销选择
                    row[i][x] = false;
                    col[j][x] = false;
                    block[i / 3 * 3 + j / 3][x] = false;
                }
            }
            false
        } else {
            true
        }
    }
    pub fn permute_unique(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        let len = nums.len();
        let mut res = vec![];
        if len == 0 {
            return res;
        }
        nums.sort();
        let mut used = vec![false; len];
        let mut path = Vec::with_capacity(len);
        fn dfs(
            nums: &mut Vec<i32>,
            len: i32,
            depth: i32,
            used: &mut Vec<bool>,
            path: &mut Vec<i32>,
            res: &mut Vec<Vec<i32>>,
        ) {
            if depth == len {
                res.push(path.clone());
                return;
            }
            for i in 0..len {
                if *used.get(i as usize).unwrap() {
                    continue;
                }
                if i > 0
                    && nums.get(i as usize) == nums.get((i - 1) as usize)
                    && !(*used.get((i - 1) as usize).unwrap())
                {
                    continue;
                }
                path.push(*nums.get(i as usize).unwrap());
                *used.get_mut(i as usize).unwrap() = true;
                dfs(nums, len, depth + 1, used, path, res);
                *used.get_mut(i as usize).unwrap() = false;
                path.pop();
            }
        }
        dfs(&mut nums, len as i32, 0, &mut used, &mut path, &mut res);
        res
    }
    // 逆序中序遍历
    pub fn convert_bst(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        let mut sum = 0;
        fn dfs(node: Option<Rc<RefCell<TreeNode>>>, num: &mut i32) {
            if let Some(root) = node {
                dfs(root.borrow().right.clone(), num);
                *num += root.borrow().val;
                root.borrow_mut().val = *num;
                dfs(root.borrow().left.clone(), num);
            }
        }
        dfs(root.clone(), &mut sum);
        root
    }
    pub fn merge_trees(
        t1: Option<Rc<RefCell<TreeNode>>>,
        t2: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        None
    }
}
