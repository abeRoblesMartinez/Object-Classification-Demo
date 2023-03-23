//
//  ViewController.swift
//  Object Classification Demo
//
//  Created by Abraham on 3/16/23.
//

import UIKit

class ViewController: UIViewController {
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    
    
    
    @IBAction func didTapStart(_ sender: UITapGestureRecognizer) {
        if let tappedView = sender.view {
            performSegue(withIdentifier: "photoSegue", sender: tappedView)
        }
    }
    
}

